subset=100
disable_condvec=0
dataset=adult
verbose=0

if [[ ${verbose} -eq 1 ]]
then
    folder=losses/verbose
else
    folder=losses/no_verbose
fi

# "baseline" "wass_gan" "cond"
# "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info"
# "exp_tablegan_info" "wass_gan+exp_tablegan_info" "wass_gan+cond+exp_tablegan_info"

# 300 1500 5000 10000 30000
#  100 500 1000 2000 

for j in 100; do
    for i in "tablegan_info+cond"; do
        time python experiments/main.py ${j} 2000 500 1000 \
        "${folder}/${i}" ${disable_condvec} ${dataset} ${verbose}
    done
done