# Code for the NLP experiment in the paper

1. clone the transformers library: ``git clone https://github.com/huggingface/transformers``
2. checkout the specific commit code:  ``git checkout 857ccdb259b7e46c60cf86c58b7ab038c63e4d4e``
3. install the ``transformers`` library: ``pip install -e .``
4. go to the directory we would be working: ``cd examples/ner``
5. replace the ``run_ner.py`` with the version of ``run_ner.py`` in this repository
6. put the ``modifier.py`` under the ``examples/ner`` directory
7. put the pre-calculated ``token_prob_of_entity.npy`` under the ``examples/ner`` directory (the ``token_prob_of_entity.npy`` contains a matrix with shape of [4, 28996], representing the conditional probability for misc / person / organization / location entities).
8. prepare the dataset of CoNLL-2003 (download txt files from [https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003) into the ``eng`` directory)
8. run the command with 8 GPUs:  ``python3 run_ner.py --data_dir ./eng --model_type bert --model_name_or_path bert-base-cased --output_dir trade0.7 --max_seq_length 128 --num_train_epochs 10 --per_gpu_train_batch_size 32 --save_steps 750 --do_train --do_eval --do_predict --trade_off 0.7``

The script should take about 10~20 minutes to run. On my local server, the F1 score of the final output results across three runs are: 0.9116 / 0.9134 / 0.9131 .

Note: If you see exceptions like ``AttributeError: 'BertTokenizer' object has no attribute 'num_added_tokens'``, please refer to the [https://github.com/huggingface/transformers/issues/3686](https://github.com/huggingface/transformers/issues/3686) issue.



