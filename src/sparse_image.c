
#include "blobs.h"
#include <assert.h>
#include <math.h>   /* sqrt */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* abs(int) */
#include <string.h> /* memset */

/* F2PY_WRAPPER_START
    function mask_to_coo( msk, ns, nf, i, j, nnz, w )
!DOC mask_to_coo takes a mask and converts it to a list
!DOC of i,j coordinates in a sparse array coo format
!DOC returns :
!DOC    0 => success
!DOC    1 => ns out of range ; 2 => nf out of range
!DOC    3 => nnz < 1 empty mask
!DOC    4 => nnz did not match this mask
        intent(c) mask_to_coo
        intent(c)
        integer*1, intent(in)  :: msk( ns, nf )
        integer, intent(hide), depend(msk) :: ns=shape(msk,0)
        integer, intent(hide), depend(msk) :: nf=shape(msk,1)
        integer(kind=-2), intent(inout), dimension(nnz) :: i, j
        integer, intent(inout) :: w(ns)
        integer, intent(hide), depend(i) :: nnz=shape(i,0)
        integer, intent(inout) :: w(ns)
        ! returns an int
        integer :: mask_to_coo
    end function mask_to_coo
F2PY_WRAPPER_END */
int mask_to_coo(int8_t msk[], int ns, int nf, uint16_t i[], uint16_t j[],
                int nnz, int nrow[]) {
    int mi, mj, idx;
    /*  int *nrow;
      nrow = (int*) malloc(ns*sizeof(int)); */
    if ((ns < 1) || (ns > 65535))
        return 1;
    if ((nf < 1) || (nf > 65535))
        return 2;
    if (nnz < 1)
        return 3;
        /* pixels per row , 2D image */
#pragma omp parallel for private(mi, mj)
    for (mi = 0; mi < ns; mi++) {
        nrow[mi] = 0;
        for (mj = 0; mj < nf; mj++) {
            if (msk[mi * nf + mj] != 0) {
                nrow[mi]++;
            }
        }
    }
    /* cumsum */
    for (mi = 1; mi < ns; mi++) {
        nrow[mi] += nrow[mi - 1];
    }
    if (nrow[ns - 1] != nnz) {
        return 4;
    }
    /* fill in */
#pragma omp parallel for private(mi, mj, idx)
    for (mi = 0; mi < ns; mi++) {
        if (mi == 0) {
            idx = 0;
        } else {
            idx = nrow[mi - 1];
        }
        if (nrow[mi] > idx) {
            for (mj = 0; mj < nf; mj++) {
                if (msk[mi * nf + mj] != 0) {
                    i[idx] = (uint16_t)mi;
                    j[idx] = (uint16_t)mj;
                    idx++;
                }
            }
        }
    }
    /*  free(nrow); */
    return 0;
}

/*
 * https://fr.mathworks.com/help/matlab/ref/sparse.html
 * S = sparse(i,j,v,m,n,nz)
 *   i = array of index i
 *   j = array of index j
 *   v = array of values
 *   S( i(k), j(k) ) == v(k)
 *   m = dimension 0 of S
 *   n = dimension 1 of S
 *   nz = number of non-zero values
 *
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
 *  scipy.sparse.coo_matrix(arg1, shape=None, dtype=None, copy=False)
 *   arg1 = ( v, i, j ) as above
 *   shape = (m, n)     as above
 *   dtype = float/double/int, etc
 */

/* F2PY_WRAPPER_START
    function sparse_is_sorted( i, j, nnz )
!DOC sparse_is_sorted checks whether the indices in i and j of a sparse
!DOC coo format come in the order that they would appear inside an image
!DOC *  @param i, j index arrays
!DOC *  @param nnz dimension of i, j
!DOC    returns 0 for all OK
!DOC         k for first non-sorted element
!DOC         -k for first duplicate
        intent(c) sparse_is_sorted
        intent(c)
        integer(kind=-2), intent(in), dimension(nnz) :: i
        integer(kind=-2), intent(in), dimension(nnz) :: j
        integer, intent(hide), depend(i) :: nnz = shape( i, 0)
        threadsafe
        ! returns
        integer sparse_is_sorted
    end function spare_is_sorted
F2PY_WRAPPER_END */
int sparse_is_sorted(uint16_t i[], uint16_t j[], int nnz) {
    int k, es, ed;
    es = nnz + 1;
    ed = nnz + 1;
    for (k = 1; k < nnz; k++) {
        if (i[k] < i[k - 1]) { /* bad, not sorted */
            es = (k < es) ? k : es;
            continue;
        }
        if (i[k] == i[k - 1]) {    /* Same row, j must be gt prev */
            if (j[k] < j[k - 1]) { /* bad */
                es = (k < es) ? k : es;
            } else if (j[k] == j[k - 1]) {
                ed = (k < ed) ? k : ed;
            } else {
                continue;
            }
        }
    }
    if ((es == (nnz + 1)) && (ed == (nnz + 1)))
        return 0;
    if (es > ed)
        return -ed;
    else
        return es;
}

/* F2PY_WRAPPER_START
    function sparse_connectedpixels( v, i, j, nnz, threshold, labels )
!DOC sparse_connectedpixels runs the connectedpixels algorithm on
!DOC a sparse image using a supplied threshold putting labels
!DOC into labels array and returning the number of blobs found
        intent(c) sparse_connectedpixels
        intent(c)
        real, intent(in), dimension(nnz) :: v
        integer(kind=-2), intent(in), dimension(nnz) :: i
        integer(kind=-2), intent(in), dimension(nnz) :: j
        integer, intent(hide), depend(i) :: nnz = shape( i, 0)
        real threshold
        integer, intent(inout), dimension(nnz) :: labels
        ! Returns
        integer :: sparse_connectedpixels
        threadsafe
    end function sparse_connectedpixels
F2PY_WRAPPER_END */
#define NOISY 0
int sparse_connectedpixels(float *restrict v, uint16_t *restrict i,
                           uint16_t *restrict j, int nnz, float threshold,
                           int32_t *restrict labels /* nnz */
) {
    int k, p, pp, ir;
    int32_t *S, *T, np;
    /* Read k = kurrent
       p = prev */
    double start, mid, end;
    if (NOISY) {
        start = my_get_time();
        k = sparse_is_sorted(i, j, nnz);
        if (k != 0)
            return k;
    }
    pp = 0;
    p = 0;
    S = dset_initialise(16384);
    /* Main loop */
    if (NOISY)
        printf("ok to main loop\n");
    for (k = 0; k < nnz; k++) {
        labels[k] = 0;
        if (v[k] <= threshold) {
            continue;
        }
        if (k == 0)
            goto newlabel;
        /* Decide on label for this one ...
         *
         * 4 neighbors : k-1 is prev
         */
        p = k - 1; /* previous pixel, same row */
        if (((j[p] + 1) == j[k]) && (i[p] == i[k]) && (labels[p] > 0)) {
            labels[k] = labels[p];
            /* check for unions below */
        }
        if (i[k] == 0)
            goto newlabel;
        ir = i[k] - 1;
        /* pp should be on row above, on or after j-1 */
        while (ir > i[pp])
            pp++;
        /* out if nothing on row above */
        if (i[pp] == i[k])
            goto newlabel;
        /* Locate previous pixel on row above */
        while (((j[k] - j[pp]) > 1) && (i[pp] == ir))
            pp++;
        for (p = pp; j[p] <= j[k] + 1; p++) {
            if (i[p] == ir) {
                if (labels[p] > 0) {
                    // Union p, k
                    match(labels[k], labels[p], S);
                }
            } else {
                break; // not same row
            }
        }
    newlabel:
        if (labels[k] == 0)
            S = dset_new(&S, &labels[k]);
    } // end loop over data
    if (NOISY)
        mid = my_get_time();
    T = dset_compress(&S, &np);
    // renumber labels
    for (k = 0; k < nnz; k++) {
        if (labels[k] > 0) {
            /* if( T[labels[k]] == 0 ){
               printf("Error in sparse_connectedpixels\n");
               } */
            labels[k] = T[labels[k]];
        }
    }
    free(S);
    free(T);
    if (NOISY) {
        end = my_get_time();
        printf("Time in sparse image %f ms %f ms\n", 1000 * (end - mid),
               1000 * (mid - start));
    }
    return np;
}

/* F2PY_WRAPPER_START
    function sparse_connectedpixels_splat( v, i, j, nnz, th, lbl, Z,ni, nj )
!DOC sparse_connectedpixels_splat is for debugging/timing. It splats
!DOC the sparse array into a dense array and runs the old connectedpixels
!DOC code on that.
        intent(c) sparse_connectedpixels_splat
        intent(c)
        real, intent(in), dimension(nnz) :: v
        integer(kind=-2), intent(in), dimension(nnz) :: i
        integer(kind=-2), intent(in), dimension(nnz) :: j
        integer, intent(hide), depend(i) :: nnz = shape( i, 0)
        real th
        integer, intent(inout), dimension(nnz) :: lbl
        integer, intent(inout), dimension((ni+2)*(nj+2)) :: Z
        integer :: ni, nj
        ! Returns
        integer :: sparse_connectedpixels_splat
        threadsafe
    end function sparse_connectedpixels_splat
F2PY_WRAPPER_END */
#define NOISY 0
int sparse_connectedpixels_splat(float *restrict v, uint16_t *restrict i,
                                 uint16_t *restrict j, int nnz, float threshold,
                                 int32_t *restrict labels, /* nnz */
                                 int32_t *restrict Z,
                                 /* workspace, at least (imax+2)*(jmax+2) */
                                 int imax, int jmax) {
    int k, p, pp, ir, jdim, ik, jk;
    int32_t *S, *T, np;
    /* Read k = kurrent
       p = prev */
    double start, mid;
    if (NOISY) {
        start = my_get_time();
        k = sparse_is_sorted(i, j, nnz);
        if (k != 0)
            return k;
    }
    if (NOISY) {
        mid = my_get_time();
        printf("check sorted %.3f ms\n", (mid - start) * 1000);
        start = my_get_time();
    }

    jdim = jmax + 2;
    /* This is not! delivered with zeros, we put a border in too
     *  Z = (int32_t *) malloc(idim*jdim* sizeof(int32_t));
     * later we will write into Z as a scratch area for labels (filled at very
     * end) */
    pp = 0;
    p = 0;
    S = dset_initialise(16384);
    if (NOISY) {
        mid = my_get_time();
        printf("mallocs %.3f ms\n", (mid - start) * 1000);
        start = my_get_time();
    }
    /* zero the parts of Z that we will read from (pixel neighbors) */
    for (k = 0; k < nnz; k++) {
        ik = i[k] + 1; /* the plus 1 is because we padded Z */
        jk = j[k] + 1;
        p = ik * jdim + jk;
        Z[p] = 0;
        Z[p - 1] = 0;
        Z[p - jdim - 1] = 0;
        Z[p - jdim] = 0;
        Z[p - jdim + 1] = 0;
    }
    if (NOISY) {
        mid = my_get_time();
        printf("zeros %.3f ms\n", (mid - start) * 1000);
        start = my_get_time();
    }

    /* Main loop */
    for (k = 0; k < nnz; k++) {
        if (v[k] <= threshold) {
            continue;
        }
        /* Decide on label for this one ...
         *
         * 4 neighbors : k-1 is prev
         */
        ik = i[k] + 1; /* the plus 1 is because we padded Z */
        jk = j[k] + 1;
        p = ik * jdim + jk;
        /* previous pixel, same row */
        if (Z[p - 1] > 0) {
            Z[p] = Z[p - 1];
        }
        /* 3 pixels on previous row */
        ir = (ik - 1) * jdim + jk;
        for (pp = ir - 1; pp <= ir + 1; pp++) {
            if (Z[pp] > 0) {
                // Union p, k
                match(Z[p], Z[pp], S);
            }
        }
        if (Z[p] == 0)
            S = dset_new(&S, &Z[p]);
    } // end loop over data
    if (NOISY) {
        mid = my_get_time();
        printf("main loop %.3f ms\n", (mid - start) * 1000);
        start = my_get_time();
    }
    T = dset_compress(&S, &np);
    // renumber labels
    for (k = 0; k < nnz; k++) {
        ik = i[k] + 1; /* the plus 1 is because we padded Z */
        jk = j[k] + 1;
        p = ik * jdim + jk;
        if (Z[p] > 0) {
            labels[k] = T[Z[p]];
        }
    }
    if (NOISY) {
        mid = my_get_time();
        printf("Relabelling %f ms\n", 1000 * (mid - start));
        start = my_get_time();
    }
    free(S);
    free(T);
    if (NOISY) {
        mid = my_get_time();
        printf("Free %f ms\n", 1000 * (mid - start));
    }
    return np;
}

/* F2PY_WRAPPER_START
    subroutine sparse_blob2Dproperties( v, i, j, nnz, labels, results, npk)
        intent(c) sparse_blob2Dproperties
!DOC sparse_blob2Dproperties fills the array results with properties of
!DOC each labelled object described by v and labels (pixel values and blob)
!DOC and positions i,j in the image.
!DOC results are:
!DOC   results[ipk,s2D_1]   = sum(1), number of pixels
!DOC   results[ipk,s2D_I]   = sum (I), total intensity
!DOC   results[ipk,s2D_fI]  = sum (f*I), intensity weighted fast index
!DOC   results[ipk,s2D_sI]  = sum (s*I), intensity weighted slow index
!DOC   results[ipk,s2D_ffI] = sum (f*f*I), intensity weighted fast^2 index
!DOC   results[ipk,s2D_sfI] = sum (s*f*I), intensity weighted slow*fast index
!DOC   results[ipk,s2D_ssI] = sum (s*s*I), intensity weighted slow^2 index
        intent(c)
        real, intent(in), dimension(nnz) :: v
        integer(kind=-2), intent(in), dimension(nnz) :: i
        integer(kind=-2), intent(in), dimension(nnz) :: j
        integer, intent(hide), depend(v) :: nnz = shape( v, 0)
        integer, intent(in), dimension(nnz) :: labels
        integer, intent(in) :: npk
        double precision, intent(out) :: results( npk, NPROPERTY2D )
        threadsafe
    end subroutine sparse_blob2Dproperties
F2PY_WRAPPER_END */
/* blob_properties for sparse - in image only... */
void sparse_blob2Dproperties(float *restrict data, uint16_t *restrict i,
                             uint16_t *restrict j, int nnz,
                             int32_t *restrict labels, double *restrict res,
                             int32_t npk) {
    int k, kpk, f, s;
    double fval;
    /* init to zero */
    for (k = 0; k < npk * NPROPERTY2D; k++) {
        res[k] = 0.0;
    }
    for (k = 0; k < npk; k++) {
        res[k * NPROPERTY2D + s2D_bb_mn_f] = 65534.;
        res[k * NPROPERTY2D + s2D_bb_mn_s] = 65534.;
    }
    /*  printf("nnz : %d\n",nnz); */
    for (k = 0; k < nnz; k++) {
        if (labels[k] == 0) {
            continue; /* background pixel */
        }
        if (labels[k] > npk) {
            printf("Error,k %d,labels[k] %d, npk %d \n", k, labels[k], npk);
        }
        kpk = (labels[k] - 1) * NPROPERTY2D;
        fval = (double)data[k];
        s = (int)i[k];
        f = (int)j[k];
        res[kpk + s2D_1] += 1.;
        res[kpk + s2D_I] += fval;
        res[kpk + s2D_fI] += (fval * f);
        res[kpk + s2D_sI] += (fval * s);
        res[kpk + s2D_ffI] += (fval * f * f);
        res[kpk + s2D_sfI] += (fval * s * f);
        res[kpk + s2D_ssI] += (fval * s * s);

        if (res[kpk + s2D_bb_mx_s] < s)
            res[kpk + s2D_bb_mx_s] = s;
        if (res[kpk + s2D_bb_mx_f] < f)
            res[kpk + s2D_bb_mx_f] = f;
        if (res[kpk + s2D_bb_mn_s] > s)
            res[kpk + s2D_bb_mn_s] = s;
        if (res[kpk + s2D_bb_mn_f] > f)
            res[kpk + s2D_bb_mn_f] = f;
    }
}

/* F2PY_WRAPPER_START
    function sparse_localmaxlabel( v, i, j, nnz, MV, iMV, labels)
        intent(c) sparse_localmaxlabel
!DOC sparse_localmaxlabel assigns labels to sparse array in sorted coo format
!DOC supplied in (v,(i,j)). MV and iMV are temporaries.
!DOC single threaded
        intent(c)
        real, intent(in), dimension(nnz) :: v
        integer(kind=-2), intent(in), dimension(nnz) :: i
        integer(kind=-2), intent(in), dimension(nnz) :: j
        integer, intent(hide), depend(v) :: nnz = shape( v, 0)
        real, intent(inout) :: MV( nnz )
        integer, intent(inout) :: iMV( nnz )
        integer, intent(inout) :: labels( nnz )
        ! returns
        integer ::  sparse_localmaxlabel
        threadsafe
    end function sparse_localmaxlabel
F2PY_WRAPPER_END */
#define CHECKSANITY 0
int sparse_localmaxlabel(float *restrict v, uint16_t *restrict i,
                         uint16_t *restrict j, int nnz,
                         float *restrict MV, // neighbor Max Val (of 3x3 square)
                         int32_t *restrict iMV,   // Which neighbor is higher?
                         int32_t *restrict labels // Which neighbor is higher?
) {
    int k, p, pp, ir;
    /* Read k = kurrent
       p = prev */
    if (NOISY) {
        k = sparse_is_sorted(i, j, nnz);
        if (k != 0) {
            printf("Not sorted! k=%d\n", k);
        }
    }
    /* prev row */
    pp = 0;
    p = 0;
    /* First pixel -  we assume it is a max, it will be stolen later
       has no previous...*/
    iMV[0] = 0;
    MV[0] = v[0];
    /* Main loop */
    for (k = 1; k < nnz; k++) {
        iMV[k] = k;   /* iMV[k] == k tags a max */
        MV[k] = v[k]; /* MV[k] is value of that max - a temporary */
        /* previous row first */
        ir = ((int)i[k]) - 1;
        /* pp should be on row above, on or after j-1 */
        while (ir > i[pp]) {
            pp++;
            if (CHECKSANITY) {
                assert((pp >= 0) && (pp < nnz));
            }
        }
        /* skip if nothing on row above */
        if (i[pp] < i[k]) {
            /* Locate previous pixel on row above */
            while (((j[k] - j[pp]) > 1) && (i[pp] == ir)) {
                pp++;
                if (CHECKSANITY) {
                    assert((pp >= 0) && (pp < nnz));
                }
            }
            /* Now the 3 pixels on the row above, if they are present */
            for (p = pp; j[p] <= j[k] + 1; p++) {
                if (CHECKSANITY) {
                    assert((p >= 0) && (p < nnz));
                }
                if (i[p] != ir)
                    break;
                if (v[k] > v[p]) { /* This one is higher */
                    /* Steal if we are higher than neighbor currently points to
                     */
                    if (v[k] > MV[p]) {
                        iMV[p] = k;
                        MV[p] = v[k];
                    }
                } else {
                    if (v[p] > MV[k]) {
                        iMV[k] = p;
                        MV[k] = v[p];
                    }
                }
            } /* 3 previous */
        }     /* row above */
        /* 4 preceding neighbors : k-1 is prev */
        p = k - 1;
        if (CHECKSANITY) {
            assert((p >= 0) && (p < nnz));
        }
        if ((i[k] == i[p]) &&
            (j[k] == (j[p] + 1))) { /* previous pixel, same row */
            if (v[k] > v[p]) {      /* This one is higher */
                /* Steal if we are higher than neighbor currently points to */
                if (v[k] > MV[p]) {
                    iMV[p] = k;
                    MV[p] = v[k];
                }
            } else if (v[p] > MV[k]) { /* Previous one was higher */
                iMV[k] = p;
                MV[k] = v[p];
            }
        }
    } // end loop over data
    /* Count max values and assign unique labels */
    pp = 0;
    for (k = 0; k < nnz; k++) {
        labels[k] = -1;
        if (iMV[k] == k) {
            pp = pp + 1; /* Labels start at one */
            labels[k] = pp;
        }
    }
    /* Now make all labels point to their root */
    for (k = 0; k < nnz; k++) {
        p = iMV[k];
        while (iMV[p] != p) {
            p = iMV[p];
            if (CHECKSANITY) {
                assert((p >= 0) && (p < nnz));
            }
        }
        labels[k] = labels[p];
    }
    return pp;
}

/* F2PY_WRAPPER_START
    function sparse_overlaps( i1, j1, k1, nnz1, &
                               i2, j2, k2, nnz2 )
!DOC sparse_overlaps identifies the pixels in i1,j1 which overlap i2,j2.
!DOC The list of overlaps is returned in k1/k2 such that i1[k1]==i2[k2]
!DOC and j1[k1]==j2[k2]. Probably assumes that sparse_is_sorted was true.
        intent(c) sparse_overlaps
        intent(c)
        integer(kind=-2), dimension(nnz1), intent(c) :: i1, j1
        integer, intent(inout), dimension(nnz1) :: k1
        integer, intent(hide), depend(i1), intent(c) :: nnz1=shape(i1,0)
        integer(kind=-2), dimension(nnz2), intent(c) :: i2, j2
        integer, intent(inout), dimension(nnz2) :: k2
        integer, intent(hide), depend(i2), intent(c) :: nnz2=shape(i2,0)
        ! returns
        integer :: sparse_overlaps
        threadsafe
    end function sparse_overlaps
F2PY_WRAPPER_END */
int sparse_overlaps(uint16_t *restrict i1, uint16_t *restrict j1,
                    int *restrict k1, int nnz1, uint16_t *restrict i2,
                    uint16_t *restrict j2, int *restrict k2, int nnz2

) {
    /*
     * Identify the overlapping pixels that are common to both
     *   i1[k1]==i2[k2] ; j1[k1]==j2[k2];
     *   fill in k1/k2
     */
    int p1, p2, nhit;
    p1 = 0;
    p2 = 0;
    nhit = 0;
    while ((p1 < nnz1) && (p2 < nnz2)) {
        /* Three cases:
         * k1 and k2 are the same pixel or one or the other is ahead */
        if (i1[p1] > i2[p2]) {
            p2++;
        } else if (i1[p1] < i2[p2]) {
            p1++;
        } else { /* i1==i2 */
            if (j1[p1] > j2[p2]) {
                p2++;
            } else if (j1[p1] < j2[p2]) {
                p1++;
            } else { /* i1=i2,j1=j2 */
                k1[nhit] = p1;
                k2[nhit] = p2;
                nhit++;
                p1++;
                p2++;
            }
        }
    }
    for (p1 = nhit; p1 < nnz1; p1++)
        k1[p1] = 0;
    for (p2 = nhit; p2 < nnz2; p2++)
        k2[p2] = 0;
    return nhit;
}

/* F2PY_WRAPPER_START
    function compress_duplicates( i, j, oi, oj, tmp, n, nt )
!DOC compress_duplicates removes duplicate i,j labels. On entry then
!DOC i and j are set as the labels from two images. They are sorted
!DOC and on exit i,j hold the unique pairs and oi holds the count
!DOC for the number of overlaps. oj and tmp are temporaries.
        intent(c) compress_duplicates
        intent(c)
        integer, dimension(n), intent(c) :: i, j
        integer, dimension(n), intent(c, inout) :: oi, oj
        integer, dimension(nt), intent(c, inout) :: tmp
        integer, intent(hide), depend(i) :: n = shape(i,0)
        integer, intent(hide), depend(tmp) :: nt = shape(tmp,0)
        ! returns
        integer :: compress_duplicates
        threadsafe
    end function compress_duplicates
F2PY_WRAPPER_END */
int compress_duplicates(int *restrict i, int *restrict j, int *restrict oi,
                        int *restrict oj, int *restrict tmp, int n, int nt) {
    int k, vmax, c, t, ik, jk;
    /* First sort on j */
    vmax = i[0];
    for (k = 0; k < n; k++) { /* length of histogram */
        if (i[k] > vmax)
            vmax = i[k];
        if (j[k] > vmax)
            vmax = j[k];
    }
    assert(vmax < nt);
    for (k = 0; k <= vmax; k++) { /* Zero the histogram */
        tmp[k] = 0;
    }
    for (k = 0; k < n; k++) { /* Make the histogram */
        tmp[j[k]] = tmp[j[k]] + 1;
    }
    c = 0;
    for (k = 0; k <= vmax; k++) { /* Cumsum */
        t = tmp[k];
        tmp[k] = c;
        c = c + t;
    }
    for (k = 0; k < n; k++) { /* Now the order is: */
        oi[tmp[j[k]]] = i[k];
        oj[tmp[j[k]]] = j[k];
        tmp[j[k]]++;
    }
    /* Now sort on i */
    for (k = 0; k <= vmax; k++) { /* Zero the histogram */
        tmp[k] = 0;
    }
    for (k = 0; k < n; k++) { /* Make the histogram */
        tmp[i[k]]++;
    }
    c = 0;
    for (k = 0; k <= vmax; k++) { /* Cumsum */
        t = tmp[k];
        tmp[k] = c;
        c = c + t;
    }
    for (k = 0; k < n; k++) { /* Now the order is: */
        /* t = order to read the original array to get sorted on j */
        j[tmp[oi[k]]] = oj[k];
        i[tmp[oi[k]]] = oi[k];
        tmp[oi[k]]++;
    }
    /* init */
    ik = i[0];
    jk = j[0];
    t = 1; /* nhits */
    c = 0; /* write pos */
    for (k = 1; k < n; k++) {
        if ((ik == i[k]) && (jk == j[k])) {
            t++; /* add one */
        } else {
            /* write prev */
            i[c] = ik;
            j[c] = jk;
            oi[c] = t;
            /* init next */
            c++;
            t = 1;
            ik = i[k];
            jk = j[k];
        }
    }
    /* write last */
    i[c] = ik;
    j[c] = jk;
    oi[c] = t;
    c++;
    return c;
}
