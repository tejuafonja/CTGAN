subset=500
folder=losses/no_verbose
test_size=10000
metric=f1+hist

# "baseline" "wass_gan" "cond"
# "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info"
# "exp_tablegan_info" "wass_gan+exp_tablegan_info" "wass_gan+cond+exp_tablegan_info"

for i in "baseline" "wass_gan" "cond" "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info" "exp_tablegan_info" "wass_gan+exp_tablegan_info" "wass_gan+cond+exp_tablegan_info"; do
    python experiments/evaluate.py ${subset} ${test_size} 1000 ${metric} ${folder}/${i}
done