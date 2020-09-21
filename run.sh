DATASETS=("cora")
TIMES=(0.01)
ACCS=(0.7)

for DATASET in $DATASETS
do
    echo "DATASET: " $DATASET
    for TIME in $TIMES
    do
        echo "TIME CONSTRAINT: " $TIME
        python BO.py --dataset "$DATASET" --time --time_constraint $TIME
    done

    for ACC in $ACCS
    do
        echo "ACC CONSTRAINT: " $ACC
        python BO.py --dataset "$DATASET" --accuracy --accuracy_constraint $ACC
    done
done
