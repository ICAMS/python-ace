#include <cmath>
#include <iostream>

#include "ace_spherical_polar.h"
#include <sstream>
/**
Constructor for SHarmonics. Dynamically initialises all the arrays.

@param lmax, int

The value of lmax

@returns None
*/
ACESHarmonics::ACESHarmonics(LS_TYPE lm) {
    init(lm);
}

void ACESHarmonics::init(LS_TYPE lm) {
    lmax = lm;

    //calculate the total size required
    //int s = (lmax+1)*(lmax+2)/2;


    alm.init(lmax, "alm");
    blm.init(lmax, "blm");
    clm.init(lmax, "clm");
    dl.init(lmax + 1);
    el.init(lmax + 1);

    plm.init(lmax, "plm");
    splm.init(lmax, "splm");
    dplm.init(lmax, "dplm");

    ylm.init(lmax, "ylm");
    dylm.init(lmax, "dylm");


    pre_compute();
}

/**
Destructor for SHarmonics. Frees the memory of all the arrays.CC = g++

@param None

@returns None
*/
ACESHarmonics::~ACESHarmonics() {
}


/**
Precomputes the value of \f$ a_{lm}, b_{lm} \f$ values. See description of data members and
@link https://arxiv.org/pdf/1410.1748.pdf @endlink for more information.

@param None

@returns None
*/
void ACESHarmonics::pre_compute() {
    //int testl = 10;

    //cout<<1<<" "<<splm[loc(1,1)]<<endl;

    DOUBLE_TYPE a, b, c;
    DOUBLE_TYPE lsq, ld, l1, l2, l3, l4;
    DOUBLE_TYPE msq;


    for (LS_TYPE l = 0; l <= lmax; l++) {

        lsq = l * l;
        ld = 2 * l;
        l1 = (4 * lsq - 1);
        l2 = lsq - ld + 1;
        l3 = ld + 1;
        l4 = ld - 1;

        for (MS_TYPE m = 0; m < l - 1; m++) {
            //calculate the a and b vals
            msq = m * m;
            a = sqrt(l1 / (lsq - msq));
            b = -sqrt((l2 - msq) / (4 * l2 - 1));
            c = sqrt(((l - m) * (l + m) * l3) / l4);

            alm(l, m) = a;
            blm(l, m) = b;
            clm(l, m) = c;


        }

    }


    for (LS_TYPE l = 2; l <= lmax; l++) {

        dl(l) = sqrt(2 * (l - 1) + 3);
        el(l) = sqrt(1 + 0.5 / DOUBLE_TYPE(l));

    }
}


/**
Function that computes \f$ P_{lm} \f$ for the corresponding lmax value
Input is the \f$ \cos(\theta) \f$ value which can be calculated from an atom
and its neighbors.

For each \f$ \cos(\theta) \f$, this computes the whole range of \f$ P_{lm} \f$ values
and its derivatives upto the lmax specified, which is a member of the class.

@param costheta, DOUBLE_TYPE

@returns None
*/
void ACESHarmonics::compute_plm(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta) {

    DOUBLE_TYPE temp = 0.39894228040143267794; //this is sqrt(0.5/pi)
    DOUBLE_TYPE sq3o1 = 1.732050808; //sqrt(3.0)
    DOUBLE_TYPE sq3o2 = 1.224744871; //sqrt(1.5)
    DOUBLE_TYPE t0d = 0.0;
    DOUBLE_TYPE t0s = 0.0;
    DOUBLE_TYPE t, td, ts;
    DOUBLE_TYPE invst;  //  sintheta^-1
    DOUBLE_TYPE plmlm1lm1;

    /*----------------------------------------------------
    calculate l=0, m=0
    ----------------------------------------------------*/
    //normal plm and m*plm/sintheta
    plm(0, 0) = temp;
    splm(0, 0) = t0s;
    dplm(0, 0) = t0d;


    //calculate l=1, m=0
    if (lmax > 0) {
        t = temp * costheta * sq3o1;
        plm(1, 0) = t;
        splm(1, 0) = t0s;
        dplm(1, 0) = temp * sq3o1 * -sintheta;


        //calculate loop l=1, m=1
        splm(1, 1) = temp * -sq3o2;
        temp = temp * sintheta * -sq3o2;
        plm(1, 1) = temp;
        t0d = (plm(0, 0) * costheta + sintheta * t0d) * -sq3o2;
        dplm(1, 1) = t0d;
    }

    //calculation of sintheta^-1
    invst = 1. / sintheta;

    for (LS_TYPE l = 2; l <= lmax; l++) {

        MS_TYPE m = 0;
        t = alm(l, m) * (costheta * plm(l - 1, m) + blm(l, m) * plm(l - 2, m));
        td = (l * costheta * t - clm(l, m) * plm(l - 1, m)) * invst;
        ts = t0s;
        plm(l, m) = t;
        splm(l, m) = ts;
        dplm(l, m) = td;

        for (MS_TYPE m = 1; m < l - 1; m++) {

            t = alm(l, m) * (costheta * plm(l - 1, m) + blm(l, m) * plm(l - 2, m));
            ts = alm(l, m) * (costheta * splm(l - 1, m) + blm(l, m) * splm(l - 2, m));
            td = (l * costheta * t - clm(l, m) * plm(l - 1, m)) * invst;

            plm(l, m) = t;
            splm(l, m) = ts;
            dplm(l, m) = td;

        }

        plm(l, l - 1) = dl(l) * costheta * temp;
        dplm(l, l - 1) = dl(l) * (-sintheta * temp + costheta * t0d);
        splm(l, l - 1) = temp * plm(l, l - 1) * (l - 1);
        splm(l, l - 1) = splm(l, l - 1) / (costheta * t0d - dplm(l, l - 1) / dl(l));

        //DOUBLE_TYPE tt = temp;
        ts = -el(l) * l * temp;
        t0d = -el(l) * (costheta * temp + sintheta * t0d);
        temp = -el(l) * sintheta * temp;

        plm(l, l) = temp;
        splm(l, l) = ts;
        dplm(l, l) = t0d;


    }

    //end of loop
}  //end compute_plm


void ACESHarmonics::compute_plm_2(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta) {
    //TODO: implement three -terms formula for sin theta = 0 case
    DOUBLE_TYPE temp = 0.39894228040143267794; //this is sqrt(0.5/pi)
    DOUBLE_TYPE sq3o1 = 1.732050808; //sqrt(3.0)
    DOUBLE_TYPE sq3o2 = 1.224744871; //sqrt(1.5)
    //takes argument as costheta
    //but when we have angles/bonds - we calculate directly
    //first declare p00 value
    //we will push it to the plm vector later on
    //these could be redefined to improve the performance slightly
    double t0 = temp; //this is sqrt(0.5/pi)
    double t0d = 0.00;
    double t, td, ts, tt;
    DOUBLE_TYPE t0s = 0.0;

    /*----------------------------------------------------
    calculate l=0, m=0
    ----------------------------------------------------*/

    //normal plm
    plm(0, 0) = t0;
    splm(0, 0) = t0s;
    dplm(0, 0) = t0d;


    //calculate l=1, m=0
    if (lmax > 0) {
        plm(1, 0) = temp * costheta * sq3o1;
        splm(1, 0) = t0s;
        td = t0 * sq3o1 * -sintheta;
        dplm(1, 0) = td;

        //calculate loop l=1, m=1
        temp = -sq3o2 * sintheta * temp;
        plm(1, 1) = temp;

        splm(1, 1) = -sq3o2 * t0;

        t0d = -sq3o2 * t0 * costheta;
        dplm(1, 1) = t0d;
    }

    //now initial round is over - run loop to find all'
    //m=0 to m<l-1 using the prefactors that were calculated
    //for each l from l=2 onwards
    for (int l = 2; l <= lmax; l++) {
        //first process m=0

        int m = 0;

        t = alm(l, 0) * (costheta * plm(l - 1, 0) + blm(l, 0) * plm(l - 2, 0));
        // == dt(prev.line)/dTheta
        td = alm(l, 0) * (-sintheta * plm(l - 1, 0) + dplm(l - 1, 0) * costheta + blm(l, 0) * dplm(l - 2, 0));
        ts = t0s;

        plm(l, 0) = t;
        splm(l, 0) = ts;
        dplm(l, 0) = td;

        //now do m=1 to l-1(not included)
        for (m = 1; m < l - 1; m++) {
            t = alm(l, m) * (costheta * plm(l - 1, m) + blm(l, m) * plm(l - 2, m));
            ts = alm(l, m) * (costheta * splm(l - 1, m) + blm(l, m) * splm(l - 2, m));
            td = alm(l, m) *
                 (-sintheta * plm(l - 1, m) + dplm(l - 1, m) * costheta + blm(l, m) * dplm(l - 2, m));

            plm(l, m) = t;

            //same as for t = , but with splm instead of plm
            splm(l, m) = ts;

            // derivative d/dTheta of prev.line
            dplm(l, m) = td;

        }

        //now missing values are for m = l-1 and m=l
        //once we do that, its over
        //original implementation
        plm(l, l - 1) = dl(l) * costheta * plm(l - 1, l - 1);
        dplm(l, l - 1) = dl(l) * (-sintheta * plm(l - 1, l - 1) + costheta * dplm(l - 1, l - 1));
        splm(l, l - 1) = (l - 1) * dl(l) * costheta * plm(l - 1, l - 1) * plm(l - 1, l - 1);
        //cout<<"before splm(l, l - 1)=("<<l<<","<<l-1<<")="<<splm(l, l - 1)<<endl;
        //cout<<"splm denom = "<<(costheta * dplm(l - 1, l - 1) - dplm(l, l - 1) / dl(l))<<endl;
        if (splm(l, l - 1) != 0) {
            DOUBLE_TYPE denom = (costheta * dplm(l - 1, l - 1) - dplm(l, l - 1) / dl(l));
            if (denom != 0)
                splm(l, l - 1) /= denom;
            else {
                stringstream s;
                s << "Mathematical error: for splm(" << l << "," << l - 1 << ") ~ 1/0 = inf. Stopping" << endl;
                throw overflow_error(s.str());
            }
        }
        //cout<<"splm(l, l - 1)=("<<l<<","<<l-1<<")="<<splm(l, l - 1)<<endl;
        plm(l, l) = -el(l) * sintheta * plm(l - 1, l - 1);
        dplm(l, l) = -el(l) * (costheta * plm(l - 1, l - 1) + sintheta * dplm(l - 1, l - 1));
        splm(l, l) = -el(l) * l * plm(l - 1, l - 1);

        // implementation from compute_plm
//        plm(l, l - 1) = dl(l) * costheta * temp;
//        dplm(l, l - 1) = dl(l) * (-sintheta * temp + costheta * t0d);
//        splm(l, l - 1) = temp * plm(l, l - 1) * (l - 1);
//        splm(l, l - 1) = splm(l, l - 1) / (costheta * t0d - dplm(l, l - 1) / dl(l));
//
//        ts = -el(l) * l * temp;
//        t0d = -el(l) * (costheta * temp + sintheta * t0d);
//        temp = -el(l) * sintheta * temp;
//
//        plm(l, l) = temp;
//        splm(l, l) = ts;
//        dplm(l, l) = t0d;


    }

} //end compute_plm_2

/**
Function that computes \f$ Y_{lm} \f$ for the corresponding lmax value
Input is the \f$ \cos(\theta) \f$ and \f$ \phi \f$ value which can be calculated from an atom
and its neighbors.

Each \f$ Y_{lm} \f$ value is a ACEComplex object with real and imaginary parts. This function also
finds the derivatives, which are stored in the Dycomponent class, with each component being a
ACEComplex object.

@param costheta, DOUBLE_TYPE
@param sintheta, DOUBLE_TYPE
@param cosphi, DOUBLE_TYPE
@param sinphi, DOUBLE_TYPE
@param lmaxi, int

@returns None
*/
void ACESHarmonics::compute_ylm(DOUBLE_TYPE costheta, DOUBLE_TYPE sintheta, DOUBLE_TYPE cosphi, DOUBLE_TYPE sinphi,
                                LS_TYPE lmaxi) {


    //compute plm
    if (abs(sintheta) > EPS)
        compute_plm(costheta, sintheta);
    else
        compute_plm_2(costheta, sintheta);

    DOUBLE_TYPE real;
    DOUBLE_TYPE img;


    const DOUBLE_TYPE norm_factor = 1.0 / sqrt(2.0);
    DOUBLE_TYPE c, s;

    DOUBLE_TYPE ctcp = costheta * cosphi;
    DOUBLE_TYPE ctsp = costheta * sinphi;

    DOUBLE_TYPE c1 = 1.0, c2 = cosphi;
    DOUBLE_TYPE s1 = 0.0, s2 = -sinphi;
    const DOUBLE_TYPE two_cos_phi = 2 * cosphi;

    //m = 0
    for (LS_TYPE l = 0; l <= lmaxi; l++) {
        ylm(l, 0).real = plm(l, 0) * norm_factor;;
        ylm(l, 0).img = 0.0;


        dylm(l, 0).a[0].real = dplm(l, 0) * ctcp * norm_factor;
        dylm(l, 0).a[0].img = 0;

        dylm(l, 0).a[1].real = dplm(l, 0) * ctsp * norm_factor;
        dylm(l, 0).a[1].img = 0.0;

        dylm(l, 0).a[2].real = -dplm(l, 0) * sintheta * norm_factor;
        dylm(l, 0).a[2].img = 0;
    }


    //m>0
    for (MS_TYPE m = 1; m <= lmaxi; m++) {

//        c is cosmphi, s sinmphi
        c = two_cos_phi * c1 - c2;
        s = two_cos_phi * s1 - s2;
        s2 = s1;
        s1 = s;
        c2 = c1;
        c1 = c;


        for (LS_TYPE l = m; l <= lmaxi; l++) {

            ylm(l, m).real = norm_factor * plm(l, m) * c;
            ylm(l, m).img = norm_factor * plm(l, m) * s;

            dylm(l, m).a[0].real = (c * ctcp * dplm(l, m) + splm(l, m) * s * sinphi) * norm_factor;
            dylm(l, m).a[0].img = (s * ctcp * dplm(l, m) - splm(l, m) * c * sinphi) * norm_factor;

            dylm(l, m).a[1].real = (dplm(l, m) * c * ctsp - splm(l, m) * s * cosphi) * norm_factor;
            dylm(l, m).a[1].img = (dplm(l, m) * s * ctsp + splm(l, m) * c * cosphi) * norm_factor;

            //cout<<" value dplm " << dplm(l,m) <<endl;
            dylm(l, m).a[2].real = (-dplm(l, m) * c * sintheta) * norm_factor;
            dylm(l, m).a[2].img = (-dplm(l, m) * s * sintheta) * norm_factor;
        }
    }

    //fill-in m<0
    for (LS_TYPE l = 1; l <= lmaxi; l++) {
        for (MS_TYPE m = 1; m <= l; m++) {
            auto phase = abs((int) m) % 2 == 0 ? 1 : -1;

            //Y_l,{-m} = (-1)^m (Y_l,{m})c.c.
//            cout<<"ylm("<<l<<","<<m<<") = "<<ylm(l,m).real<<","<<ylm(l,m).img<<endl;
            ylm(l, -m) = ylm(l, m) * phase;
            //complex conjugate
            ylm(l, -m).img *= -1;
//            cout<<"ylm("<<l<<","<<-m<<") = "<<yl_neg_m(l,m).real<<","<<yl_neg_m(l,m).img<<endl;
//
//            cout<<"dylm("<<l<<","<<m<<") = ["<<"("<<dylm(l,m).a[0].real<<","<<dylm(l,m).a[0].img<<")"<<
//                                                    "("<<dylm(l,m).a[1].real<<","<<dylm(l,m).a[1].img<<")"<<
//                                                    "("<<dylm(l,m).a[2].real<<","<<dylm(l,m).a[2].img<<")"<<"]"<<endl;
            //DY_l,{-m} = (-1)^m (DY_l,{m})c.c.
            dylm(l, -m) = dylm(l, m);
            dylm(l, -m) *= phase;
            //complex conjugate
            dylm(l, -m).a[0].img *= -1;
            dylm(l, -m).a[1].img *= -1;
            dylm(l, -m).a[2].img *= -1;


//            cout<<"dylm("<<l<<","<<-m<<") = ["<<"("<<dyl_neg_m(l,m).a[0].real<<","<<dyl_neg_m(l,m).a[0].img<<")"<<
//                "("<<dyl_neg_m(l,m).a[1].real<<","<<dyl_neg_m(l,m).a[1].img<<")"<<
//                "("<<dyl_neg_m(l,m).a[2].real<<","<<dyl_neg_m(l,m).a[2].img<<")"<<"]"<<endl;
        }
    }

}