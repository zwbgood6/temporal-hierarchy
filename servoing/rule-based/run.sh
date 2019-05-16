: 'run.sh'

for i in $(seq 0 100)
do
    python3 rule_based_pusher.py --seed $i --env_name TwoObjectPush-v0 --prod --prod_ix $i | grep -v 'Found'
done
