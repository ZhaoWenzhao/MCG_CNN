# !/bin/bash

MNIST_SCALE_ROT_SHEAR_DIR="${MNIST_SCALE_ROT_SHEAR_DIR:-./datasets}"


function train_rot_scale_shear_mnist() {
    # 1 model_name 
    # 2 extra_scaling 
    for seed in {0..5}
    do 
        data_dir="$MNIST_SCALE_ROT_SHEAR_DIR/MNIST_rot_scale_shear/seed_$seed/rot_scale_shear_0.3_1.0"
        python train_rot_scale_shear_mnist.py \
            --batch_size 128 \
            --epochs 60 \
            --optim adam \
            --lr 0.01 \
            --lr_steps 20 40 \
            --model $1 \
            --save_model_path "./saved_models/mnist_rot_scale_shear/$1_extra_rot_scale_shear_$2.pt" \
            --cuda \
            --extra_scaling $2 \
            --tag "sesn_experiments" \
            --data_dir="$data_dir" \

    done               
}


model_list=(
    #"mnist_ses_rst_vector_56_rot_8_interrot_8"
    "resnext50_32x4d_rst_mcg_e"
)

for model_name in "${model_list[@]}"
do
    for extra_rot_scaling_shear in 0.5  
    do 
        train_rot_scale_shear_mnist "$model_name" "$extra_rot_scaling_shear"  
    done
done