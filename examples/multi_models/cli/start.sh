export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
torchserve --start \
          --model-store model_store \
          --models demo.mar \
          --ts-config config.properties
