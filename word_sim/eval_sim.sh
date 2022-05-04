# # dataset directory
dataset=./datasets
# # text file name; one document per line
# text_file=text.txt

# # word embedding output file name


out_file=fine_tuned_embeddings_final
emb_file=${out_file}
python sim.py --dataset ${dataset} --emb_file ${emb_file}
