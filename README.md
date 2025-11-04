source .venv/bin/activate

python custom/train.py output/users_bin.csv output/edges_weighted_bin.txt
python paper_imp/train.py output/users.csv output/edges_weighted.txt

then make sure your model is in rbert model not rber/etc
python train_classifier.py --proj_type="custom"
python infer.py --proj_type="custom"
python metrics.py --proj_type="custom"
or paper_imp


awk -F',' 'NR>1{print $1, $2, $3}' edges.csv > edges_weighted.txt
