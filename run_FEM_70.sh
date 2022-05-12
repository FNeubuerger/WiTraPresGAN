#!/bin/bash
train=true
export TZ="GMT-8"
frac=70
# Experiment variables
exp="FEM_GAN_$frac"

# Iteration variables
emb_epochs=5000
sup_epochs=5000
gan_epochs=500

python main.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              420133769 \
--feat_pred_no      1 \
--max_seq_len       367 \
--train_rate        0.5 \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        10000 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \
--data_fraction     $frac