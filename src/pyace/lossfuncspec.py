class LossFunctionSpecification:

    def __init__(self, kappa=0.0,
                 L1_coeffs=0,
                 L2_coeffs=0,
                 w0_rad=0,
                 w1_rad=0,
                 w2_rad=0,
                 w_orth=0,
                 **kwargs):
        # super(LossFunctionSpecification, self).__init__(self)
        self.kappa = kappa
        self.L1_coeffs = L1_coeffs
        self.L2_coeffs = L2_coeffs

        self.w0_rad = w0_rad
        self.w1_rad = w1_rad
        self.w2_rad = w2_rad
        self.w_orth = w_orth

    def __str__(self):
        return ("LossFunctionSpecification(kappa={kappa}, L1={L1_coeffs}, " +
                "L2={L2_coeffs}, " +
                "DeltaRad=({w0_rad}, {w1_rad}, {w2_rad}), w_orth={w_orth})").format(
            kappa=self.kappa,
            L1_coeffs=self.L1_coeffs,
            L2_coeffs=self.L2_coeffs,
            w0_rad=self.w0_rad,
            w1_rad=self.w1_rad,
            w2_rad=self.w2_rad,
            w_orth=self.w_orth
        )
