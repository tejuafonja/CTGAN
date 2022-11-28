subset=10000

# "baseline" "wass_gan" "cond"
# "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info"

for i in "baseline" "wass_gan" "cond" "l2_sdgym_info" "wass_gan+l2_sdgym_info" "wass_gan+cond+l2_sdgym_info" "exp_tablegan_info" "wass_gan+exp_tablegan_info" "wass_gan+cond+exp_tablegan_info"; do
    time python experiments/main.py ${subset} 2000 500 1000 losses/no_verbose/${i}
done