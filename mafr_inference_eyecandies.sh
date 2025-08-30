export CUDA_VISIBLE_DEVICES=0

dataset_path=datasets/eyecandies
checkpoint_savepath=checkpoints/afr_eyecandies
epochs=100
batch_size=1

quantitative_folder=results/quantitatives_eyecandies

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

for class_name in "${class_names[@]}"
    do
        python afr_inference.py --class_name $class_name --model_type $model --epochs_no $epochs --batch_size $batch_size --dataset_path $dataset_path --checkpoint_folder $checkpoint_savepath --quantitative_folder $quantitative_folder --produce_qualitatives
    done