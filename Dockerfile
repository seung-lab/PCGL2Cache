FROM gcr.io/neuromancer-seung-import/pychunkedgraph:graph-tool_dracopy

ENV GIT_SSL_NO_VERIFY=1
RUN mkdir -p /home/nginx/.cloudvolume/secrets && chown -R nginx /home/nginx && usermod -d /home/nginx -s /bin/bash nginx

COPY . /app
RUN pip install pip==20.0.1 \
    && pip install --no-cache-dir --upgrade -r requirements.txt \
    && pip install --upgrade git+https://github.com/seung-lab/PyChunkedGraph.git@master