/* plugin/spechub/ref_spechub.cpp -- the coupling ops on the host.
 *
 * One row at a time through the abi/rad_spechub.h helpers. These are the rows the device kernels
 * are checked against, and the only implementation a machine with no card can run.
 */
#include "spechub.h"

int rad_ref_mtp_couple_propose(const RadArgs* a, RadStream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t row = 0; row < c.M; ++row) rad_mtp_couple_propose_row(c, row);
    return RAD_OK;
}

int rad_ref_mtp_couple_verify(const RadArgs* a, RadStream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t row = 0; row < c.M; ++row) rad_mtp_couple_verify_row(c, row);
    return RAD_OK;
}

/* One sequence at a time through rad_dflash_couple_walk, the definition the device kernel matches
 * bit for bit. */
int rad_ref_dflash_couple(const RadArgs* a, RadStream) {
    RadDflashCoupleCall c{};
    const int rc = rad_dflash_couple_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t s = 0; s < c.M; ++s) {
        const RadMtpCoupleDraw& d =
            *reinterpret_cast<const RadMtpCoupleDraw*>(c.draw + s * c.ld_draw);
        rad_dflash_couple_walk(c.cand + s * c.steps * c.ld_cand, c.unary + s * c.steps * c.ld_unary,
                               c.ld_cand, c.ld_unary,
                               c.hp ? c.hp + s * c.steps * c.ld_hp : nullptr, c.ld_hp,
                               c.anchor[s * c.anchor_stride], c.pred, c.succ, (int)c.K, c.R,
                               c.vocab, c.steps, c.q_temp, d, d.card == c.card[s],
                               c.tokens + s * c.ld_tok, c.state + s * c.st_row, c.st_round);
    }
    return RAD_OK;
}
