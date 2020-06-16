#!/usr/bin/env bash
#set -e
# this script is based on local/multicondition/run_nnet2_common.sh
# minor corrections were made to dir names for nnet3

stage=1
snrs="20:10:15:5:0"
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_data_reps=3
base_rirs="simulated"
datasets=$@

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# check if the required tools are present
local/check_version.sh || exit 1;

mkdir -p exp/nnet3
if [ $stage -le 1 ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises

  if [ ! -e rirs_noises.zip ]; then
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  rvb_opts=()
  if [ "$base_rirs" == "simulated" ]; then
    # This is the config for the system using simulated RIRs and point-source noises
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)
  else
    # This is the config for the JHU ASpIRE submission system
    rvb_opts+=(--rir-set-parameters "1.0, RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/real_rirs_isotropic_noises/noise_list)
  fi

  # corrupt the fisher data to generate multi-condition data
  # for data_dir in train dev test; do
  for data_dir in ${datasets}; do
    if [ "$data_dir" == *train* ]; then
      num_reps=$num_data_reps
    else
      num_reps=1
    fi
    python steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs $foreground_snrs \
      --background-snrs $background_snrs \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications $num_reps \
      --max-noises-per-minute 1 \
      --source-sampling-rate 16000 \
      data/${data_dir} data/${data_dir}_rvb
  done
fi