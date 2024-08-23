/*
 * =============================================================================
 *
 *       Filename:  onsetlib.c
 *
 *        Purpose:  Routines for calculating the onset functions
 *
 *      Copyright:  2020–2024, QuakeMigrate developers.
 *        License:  GNU General Public License, Version 3
 *                  (https://www.gnu.org/licenses/gpl-3.0.html)
 *
 * =============================================================================
 */

#include "qmlib.h"

/*
 * Function: overlapping_sta_lta
 * -----------------------------
 * Compute the STA/LTA onset function with overlapping windows. The return
 * value is allocated to the last sample of the STA window.
 *
 *                                                |--- STA ---|
 *    |------------------------- LTA -------------------------|
 *                                                            ^
 *                                                Value assigned here
 *
 * Arguments
 * ---------
 * signal: Pointer to array containing pre-processed waveform data.
 * head: Pointer to a struct containing the number of elements in signal and
 *       the number of samples in the short term and long term windows.
 * onset: Pointer to output array for the STA/LTA onset function.
 */
void
overlapping_sta_lta(const double * signal,
  const stalta_header * head, double * onset) {

  int i;
  const double frac = (double) head -> nlta / (double) head -> nsta;
  double buf, sta = 0., lta;

  for (i = 0; i < head -> nsta; ++i) {
    sta += signal[i];
  }
  lta = sta;
  for (i = head -> nsta; i < head -> nlta; ++i) {
    buf = signal[i];
    lta += buf;
    sta += buf - signal[i - head -> nsta];
  }
  onset[head -> nlta - 1] = sta / lta * frac;
  for (i = head -> nlta; i < head -> n; ++i) {
    buf = signal[i];
    sta += buf - signal[i - head -> nsta];
    lta += buf - signal[i - head -> nlta];
    onset[i] = sta / lta * frac;
  }
}

/*
 * Function: centred_sta_lta
 * -----------------------------
 * Compute the STA/LTA onset function with consecutive windows. The return
 * value is allocated to the last sample of the LTA window.
 *
 *                                                             |--- STA ---|
 *    |------------------------- LTA -------------------------|
 *                                                            ^
 *                                                Value assigned here
 *
 * Arguments
 * ---------
 * signal: Pointer to array containing pre-processed waveform data.
 * head: Pointer to a struct containing the number of elements in signal and
 *       the number of samples in the short term and long term windows.
 * onset: Pointer to output array for the STA/LTA onset function.
 */
void
centred_sta_lta(const double * signal,
  const stalta_header * head, double * onset) {

  int i;
  const double frac = (double) head -> nlta / (double) head -> nsta;
  double sta = 0., lta = 0.;

  // Calculate initial LTA
  for (i = 0; i < head -> nlta; ++i) {
    lta += signal[i];
  }

  // Calculate initial STA (starting from last sample of long-term window)
  for (i = head -> nlta; i < head -> nlta + head -> nsta; ++i) {
    sta += signal[i];
  }

  // Set first value, and loop over rest of signal
  onset[head -> nlta - 1] = sta / lta * frac;
  for (i = head -> nlta; i < head -> n - head -> nsta; ++i) {
    sta += signal[i + head -> nsta] - signal[i];
    lta += signal[i] - signal[i - head -> nlta];
    if (lta > 0.) {
      onset[i] = sta / lta * frac;
    } else {
      onset[i] = 1.;
    }
  }
}

/*
 * Function: recursive_sta_lta
 * -----------------------------
 * Compute the STA/LTA onset function with consecutive windows using a
 * recursive method (minimises memory costs). Reproduces exactly the centred
 * STA/LTA onset.
 *
 * Implementation influenced by the ObsPy function of the same name.
 *
 * Arguments
 * ---------
 * signal: Pointer to array containing pre-processed waveform data.
 * head: Pointer to a struct containing the number of elements in signal and
 *       the number of samples in the short term and long term windows.
 * onset: Pointer to output array for the STA/LTA onset function.
 */
void
recursive_sta_lta(const double * signal,
  const stalta_header * head, double * onset) {

  int i;
  const double csta = 1. / (double) head -> nsta;
  const double clta = 1. / (double) head -> nlta;
  double buf, sta = 0., lta = 0.;

  for (i = 1; i < head -> n; ++i) {
    buf = signal[i];
    sta = csta * buf + (1 - csta) * sta;
    lta = clta * buf + (1 - clta) * lta;
    onset[i] = sta / lta;
  }

  // Null first nlta to remove transient signal from "recursive" measure
  if (head -> nlta < head -> n) {
    for (i = 0; i < head -> nlta; ++i) {
      onset[i] = 1.;
    }
  }
}
