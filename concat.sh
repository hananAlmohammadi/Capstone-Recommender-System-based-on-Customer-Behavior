for i in {0..15}; do
    cat file_${i}.pkl >> file.pkl
    echo "file_$i concatenated"
    rm file_${i}.pkl
    echo "file_$i removed"
done