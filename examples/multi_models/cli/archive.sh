torchserve --stop

rm src.zip
zip -r src.zip src

torch-model-archiver --model-name demo \
                     --version 1.0 \
                     --force \
                     --export-path model_store \
                     --handler src/handler.py \
                     --extra-files src.zip,saved_model/best-ckp_det.pth,saved_model/best-ckp_rec.pth,src/parse_config.py
