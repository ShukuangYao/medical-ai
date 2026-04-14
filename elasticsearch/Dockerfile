FROM elasticsearch:8.11.0

# Install IK analysis plugin (for Chinese tokenization).
# This is a free, open-source plugin; it must match the ES minor version.
RUN bin/elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/8.11.0

