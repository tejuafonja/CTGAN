subset=100
folder=losses/no_verbose
test_size=10000
metric=f1+hist
dataset=adult
disable_condvec=0

# "baseline" "wass_gan" "cond"
# "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info"
# "exp_tablegan_info" "wass_gan+exp_tablegan_info" "wass_gan+cond+exp_tablegan_info"
# 300 1500 2000 5000 30000

for j in 100; do
    for i in "wass_gan+cond" "tablegan_info+cond"; do
        python experiments/evaluate.py ${j} ${test_size} 1000 ${metric} ${folder}/${i} \
        ${disable_condvec} ${dataset}
    done
done