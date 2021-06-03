import sys
import time
import typing
import logging
import datetime
from datetime import datetime
from datetime import timedelta

import numpy as np
from multiwrapper import multiprocessing_utils as mu
from google.auth import credentials
from google.cloud import bigtable
from google.api_core.retry import Retry
from google.api_core.retry import if_exception_type
from google.api_core.exceptions import Aborted
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.bigtable.table import Table
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_filters import RowFilter
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from . import utils
from . import BigTableConfig
from ..base import SimpleClient
from ... import attributes
from ... import exceptions
from ...utils.serializers import serialize_uint64
from ...utils.serializers import deserialize_uint64
from ...meta import ChunkedGraphMeta


class Client(bigtable.Client, SimpleClient):
    def __init__(
        self,
        table_id: str,
        config: BigTableConfig = BigTableConfig(),
        graph_meta: ChunkedGraphMeta = None,
    ):
        if config.CREDENTIALS:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
                credentials=config.CREDENTIALS,
            )
        else:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
            )
        self._instance = self.instance(config.INSTANCE)
        self._table = self._instance.table(table_id)

        self.logger = logging.getLogger(
            f"{config.PROJECT}/{config.INSTANCE}/{table_id}"
        )
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.WARNING)
            self.logger.addHandler(sh)
        self._graph_meta = graph_meta

    @property
    def graph_meta(self):
        return self._graph_meta

    # BASE
    def create_table(self, meta: ChunkedGraphMeta) -> None:
        """Initialize the graph and store associated meta."""
        if not meta.graph_config.OVERWRITE and self._table.exists():
            ValueError(f"{self._table.table_id} already exists.")
        self._table.create()
        self._create_column_families()
        self.write_metadata(meta)

    def write_metadata(self, meta: ChunkedGraphMeta):
        self._graph_meta = meta
        row = self.mutate_entry(
            attributes.GraphMeta.key,
            {attributes.GraphMeta.Meta: meta},
        )
        self.write([row])

    def read_metadata(self) -> ChunkedGraphMeta:
        row = self._read_byte_row(attributes.GraphMeta.key)
        self._graph_meta = row[attributes.GraphMeta.Meta][0].value
        return self._graph_meta

    def read_entries(
        self,
        start_id=None,
        end_id=None,
        end_id_inclusive=False,
        node_ids=None,
        attributes=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
    ):
        """
        Read nodes and their attributes.
        Accepts a range of node IDs or specific node IDs.
        """
        rows = self._read_byte_rows(
            start_key=serialize_uint64(start_id) if start_id is not None else None,
            end_key=serialize_uint64(end_id) if end_id is not None else None,
            end_key_inclusive=end_id_inclusive,
            row_keys=(serialize_uint64(node_id) for node_id in node_ids)
            if node_ids is not None
            else None,
            columns=attributes,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return {
            deserialize_uint64(row_key, fake_edges=fake_edges): data
            for (row_key, data) in rows.items()
        }

    def read_entry(
        self,
        node_id: np.uint64,
        attributes: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
    ) -> typing.Union[
        typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
        typing.List[bigtable.row_data.Cell],
    ]:
        """Convenience function for reading a single node from Bigtable."""
        return self._read_byte_row(
            row_key=serialize_uint64(node_id, fake_edges=fake_edges),
            columns=attributes,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )

    def write_entries(self, nodes, root_ids=None, operation_id=None):
        """Writes/updates entries with attributes."""
        # TODO convert entries and attributes to bigtable rows
        pass

    # Helpers
    def write(
        self,
        rows: typing.Iterable[bigtable.row.DirectRow],
        root_ids: typing.Optional[
            typing.Union[np.uint64, typing.Iterable[np.uint64]]
        ] = None,
        operation_id: typing.Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk."""
        initial = 1
        if slow_retry:
            initial = 5

        exception_types = (Aborted, DeadlineExceeded, ServiceUnavailable)
        retry = Retry(
            predicate=if_exception_type(exception_types),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=self.graph_meta.graph_config.ROOT_LOCK_EXPIRY.seconds,
        )

        for i in range(0, len(rows), block_size):
            status = self._table.mutate_rows(rows[i : i + block_size], retry=retry)
            if not all(status):
                raise exceptions.ChunkedGraphError(
                    f"Bulk write failed: operation {operation_id}"
                )

    def mutate_entry(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> bigtable.row.Row:
        """Mutates a single row (doesn't write to big table)."""
        row = self._table.direct_row(row_key)
        for column, value in val_dict.items():
            row.set_cell(
                column_family_id=column.family_id,
                column=column.key,
                value=column.serialize(value),
                timestamp=time_stamp,
            )
        return row

    def get_compatible_timestamp(
        self, time_stamp: datetime, round_up: bool = False
    ) -> datetime:
        return utils.get_google_compatible_time_stamp(time_stamp, round_up=False)

    # PRIVATE METHODS
    def _create_column_families(self):
        # TODO hardcoded, not good
        f = self._table.column_family("0")
        f.create()
        f = self._table.column_family("1", gc_rule=MaxVersionsGCRule(1))
        f.create()
        f = self._table.column_family("2")
        f.create()
        f = self._table.column_family("3", gc_rule=MaxVersionsGCRule(1))
        f.create()

    def _read_byte_rows(
        self,
        start_key: typing.Optional[bytes] = None,
        end_key: typing.Optional[bytes] = None,
        end_key_inclusive: bool = False,
        row_keys: typing.Optional[typing.Iterable[bytes]] = None,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> typing.Dict[
        bytes,
        typing.Union[
            typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
            typing.List[bigtable.row_data.Cell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets."""
        # Create filters: Rows
        row_set = RowSet()
        if row_keys is not None:
            for row_key in row_keys:
                row_set.add_row_key(row_key)
        elif start_key is not None and end_key is not None:
            row_set.add_row_range_from_keys(
                start_key=start_key,
                start_inclusive=True,
                end_key=end_key,
                end_inclusive=end_key_inclusive,
            )
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or"
                " both, a start row and an end row."
            )
        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
        )
        # Bigtable read with retries
        rows = self._read(row_set=row_set, row_filter=filter_)

        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    cell_entry.value = column.deserialize(cell_entry.value)
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, attributes._Attribute):
                rows[row_key] = cell_entries
        return rows

    def _read_byte_row(
        self,
        row_key: bytes,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> typing.Union[
        typing.Dict[attributes._Attribute, typing.List[bigtable.row_data.Cell]],
        typing.List[bigtable.row_data.Cell],
    ]:
        """Convenience function for reading a single row."""
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return (
            row.get(row_key, [])
            if isinstance(columns, attributes._Attribute)
            else row.get(row_key, {})
        )

    def _execute_read_thread(self, args: typing.Tuple[Table, RowSet, RowFilter]):
        table, row_set, row_filter = args
        if not row_set.row_keys and not row_set.row_ranges:
            # Check for everything falsy, because Bigtable considers even empty
            # lists of row_keys as no upper/lower bound!
            return {}
        range_read = table.read_rows(row_set=row_set, filter_=row_filter)
        res = {v.row_key: utils.partial_row_data_to_column_dict(v) for v in range_read}
        return res

    def _read(
        self, row_set: RowSet, row_filter: RowFilter = None
    ) -> typing.Dict[
        bytes, typing.Dict[attributes._Attribute, bigtable.row_data.PartialRowData]
    ]:
        """Core function to read rows from Bigtable. Uses standard Bigtable retry logic."""
        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now
        max_row_key_count = 1000
        n_subrequests = max(1, int(np.ceil(len(row_set.row_keys) / max_row_key_count)))
        n_threads = min(n_subrequests, 2 * mu.n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[
                i * max_row_key_count : (i + 1) * max_row_key_count
            ]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges
        responses = mu.multithread_func(
            self._execute_read_thread,
            params=((self._table, r, row_filter) for r in row_sets),
            debug=n_threads == 1,
            n_threads=n_threads,
        )

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)
        return combined_response
