input_dir=$1   # ../datasets/USPTO_50K/aug20
out_dir=${input_dir}/data-bin
dict=$2  # dict.txt in current folder
workers=40
src=src
tgt=tgt
frag=frag
ref=ref
sim=sim
echo "Processing Main Task ($src -> $tgt)..."
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $input_dir/train --validpref $input_dir/val --testpref $input_dir/test \
    --destdir $out_dir \
    --workers $workers \
    --tgtdict $dict \
    --srcdict $dict \
    # --joined-dictionary \

if [ -f "$input_dir/train.$frag" ]; then
    echo "Processing Fragments ($frag)..."
    fairseq-preprocess --only-source \
        --source-lang $frag \
        --trainpref $input_dir/train --validpref $input_dir/val --testpref $input_dir/test \
        --destdir $out_dir \
        --workers $workers \
        --srcdict $dict
fi

if [ -f "$input_dir/train.$ref" ]; then
    echo "Processing References ($ref)..."
    fairseq-preprocess --only-source \
        --source-lang $ref \
        --trainpref $input_dir/train --validpref $input_dir/val --testpref $input_dir/test \
        --destdir $out_dir \
        --workers $workers \
        --srcdict $dict
fi

if [ -f "$input_dir/train.$sim" ]; then
    echo "Copying Similarity ($sim) to data-bin..."
    # 假设文件名是 train.sim, val.sim, test.sim
    cp $input_dir/train.$sim $out_dir/train.$sim
    cp $input_dir/val.$sim $out_dir/valid.$sim
    cp $input_dir/test.$sim $out_dir/test.$sim
fi

echo "All Done! Data saved to $out_dir"