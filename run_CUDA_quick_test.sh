#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="quick_test"

# Iteration variables
emb_epochs=5
sup_epochs=5
gan_epochs=5

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
--batch_size        2048 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \
