python pytorch/train.py config/config.yaml relu
python pytorch/test.py config/config.yaml relu
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

python pytorch/train.py config/config.yaml rrelu
python pytorch/test.py config/config.yaml rrelu
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

python pytorch/train.py config/config.yaml prelu
python pytorch/test.py config/config.yaml prelu
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

python pytorch/train.py config/config.yaml relu6
python pytorch/test.py config/config.yaml relu6
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

python pytorch/train.py config/config.yaml selu
python pytorch/test.py config/config.yaml selu
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json

python pytorch/train.py config/config.yaml dyrelu
python pytorch/test.py config/config.yaml dyrelu
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json