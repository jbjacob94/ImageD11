


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "blobs.h"  /* INTEGER */




/*
# ImageD11_v1.x Software for beamline ID11
# Copyright (C) 2005-2017  Jon Wright
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/


/* Updated to use f2py to generate bindings, 2017 */

static char moduledocs[] =\
"   C extensions for image analysis, part of ImageD11\n"\
"   ";


#include "blobs.h"   /* Disjoint sets thing for blob finding */

/* ==== connectedpixels ======================================== */

static char connectedpixels_doc[] =\
"   nblobs = connectedpixels ( data=Numeric.array(data, 2D)  , \n"\
"                              labels=Numeric.array(blob, 2D, Int)  ,\n"\
"                              threshold=float threshold ,\n"\
"                              verbose=Int verbose )\n"\
"   data is normally an image \n"\
"   blob is an array to receive pixel -> blob assignments\n"\
"   threshold is the value above which a pixel is considered to be in a blob\n"\
"   verbose flags printing on stdout";

                              
/* Fill in an image of peak assignments for pixels */


void match( INTEGER *new, INTEGER *old, INTEGER *S){
  /* printf("match %d %d\n",*new,*old); */
  if(*new == 0) { 
      *new = *old ; 
  } else {
    if ( *new != *old ) {
        dset_makeunion( S, *old, *new);
    }
  }
}


int connectedpixels( float* data, INTEGER* labels, float threshold, int verbose,
     int eightconnected, int ns, int nf){

  int i, j, irp, ir, ipx;
  INTEGER k, *S, *T, np;

  if( verbose ){ 
      printf("Welcome to connectedpixels ");
      if(eightconnected) printf("Using connectivity 8\n");
      else printf("Using connectivity 4\n");
  }

  /* lots of peaks possible */
  S = dset_initialise( 16384 ) ;

  /* To simplify later we hoist the first row and first pixel
   * out of the loops
   *
   * Algorithm scans image looking at stuff previously seen
   */

  /* First point */
  /*  i = 0;   j = 0; */
  if (data[0] > threshold) {
      S = dset_new( &S, &labels[0] );
  } else {
      labels[0] = 0;
  }
  /* First row */
  for (j = 1; j<nf; j++){
      labels[j] = 0; /* initialize */
      if(data[j] > threshold){
          if( labels[j-1] > 0) {
              labels[j] = labels[j-1];
          } else {
              S = dset_new( &S, &labels[j] );
          }
      }
  }

  /* === Mainloop =============================================*/
  for(i=1; i<ns;i++){ /* i-1 prev row always exists, see above */
      ir = i*nf;   /* this row */
      irp = ir-nf; /* prev row */
      /* First point */
      /* j=0; */
      labels[ir]=0;
      if( data[ir] > threshold ){
          if( labels[irp] > 0){
              labels[ir] = labels[irp];
          }
          if( eightconnected && ( labels[irp+1] > 0 ) ){
              match( &labels[ir], &labels[irp+1], S);
          }
          if( labels[ir] == 0 ){
              S = dset_new( &S, &labels[ir] );
          }
      }
      /* Run along row to just before end */
      for(j=1; j<nf-1; j++){
          ipx = ir+j;
          irp = ipx-nf;
          labels[ipx]=0;
          if( data[ipx] > threshold ){
              /* Pixel needs to be assigned */
              if( eightconnected && ( labels[irp-1] > 0 ) ){
                  match( &labels[ ipx ], &labels[irp-1], S);
              }
              if( labels[irp] > 0 ){
                  match( &labels[ ipx ], &labels[irp], S);
              }
              if( eightconnected && ( labels[irp+1] > 0 )){
                  match( &labels[ ipx ], &labels[irp+1], S);
              }
              if( labels[ipx-1] > 0){
                  match( &labels[ ipx ], &labels[ipx-1], S);
              }
              if ( labels[ipx] == 0 ){ /* Label is new ! */
                  S = dset_new( &S, &labels[ipx] );
              }
          } /* (val > threshold) */
      } /* Mainloop j */
      /* Last pixel on the row */
      ipx = ir+nf-1;  
      irp = ipx-nf;
      labels[ipx]=0;
      if( data[ipx] > threshold ){
              if( eightconnected && ( labels[irp-1] > 0 ) ){
                  match( &labels[ ipx ], &labels[irp-1], S);
              }
              if( labels[irp] > 0 ){
                  match( &labels[ ipx ], &labels[irp], S);
              }
              if( labels[ipx-1] > 0){
                  match( &labels[ ipx ], &labels[ipx-1], S);
              }
              if ( labels[ipx] == 0 ){ /* Label is new ! */
                  S = dset_new( &S, &labels[ipx] );
              }
      }




  }
  /* Now compress the disjoint set to make single list of 
   * unique labels going from 1->n
   */
  T = dset_compress( &S, &np );
  /* Now scan through image re-assigning labels as needed */
  for(i=0;i<ns;i++){
      for(j=0;j<nf;j++){
          ipx = i*nf + j;
          k = labels[ipx];
          if( k > 0 ){
             if( T[k] == 0 ) {
                 printf("Error in connectedpixels\n");
             }
             if( T[k] != k ) { 
                 labels[i*nf+j] = T[k];
             }
          }
      }
  }
  free(S);
  free(T);
  return np;
}


static char blobproperties_doc[] =\
  "res = blobproperties ( Numeric.array(data, 2D)  , \n"        \
  "                          Numeric.array(blob, 2D, Int)  ,\n"        \
  "                          Int np , \n" \
  "                          Int verbose )\n"                                \
  "\n"                                                                        \
  "   Computes various properties of a blob image "\
  "           (created by connectedpixels)\n" \
  "   data  = image data \n"                                                \
  "   blob  = integer peak assignments from connectedpixels \n"                \
  "   np    = number of peaks to treat \n"                                \
  "   verbose  - flag about whether to print\n"                                \
  "  \n";


void blobproperties(float* data, INTEGER* labels, INTEGER np, float omega,
        int verbose, 
        int ns, int nf, double* res){
   int i, j, bad, ipx;
   double fval;
   INTEGER ipk;
   if(verbose){
       printf("Computing blob moments, ns %d, nf %d, np %d\n",ns,nf,np);
   }
  /* Initialise the results */
   for ( i=0 ; i<np ; i++) {
     for ( j=0 ; j<NPROPERTY; j++){
           res[i*NPROPERTY+j]=0.;
        } 
     /* Set min to max +1 and vice versa */
     res[i*NPROPERTY+bb_mn_f]=nf+1;
     res[i*NPROPERTY+bb_mn_s]=ns+1;
     res[i*NPROPERTY+bb_mx_f]=-1;
     res[i*NPROPERTY+bb_mx_s]=-1;
     /* All pixels have the same omega in this frame */
     res[i*NPROPERTY+bb_mx_o]=omega;
     res[i*NPROPERTY+bb_mn_o]=omega;
   }
   if(verbose!=0)printf("Scanning image\n");
      
   bad = 0;
   /* i,j is looping along the indices data array */
   for( i = 0 ; i < ns ; i++ ){  
     for( j = 0 ; j < nf ; j++ ){
       ipx = i*nf+j;
       ipk = labels[ipx];
       if( ipk > 0  && ipk <=np ) {
            fval = (double) data[ipx];
            add_pixel( &res[NPROPERTY*(ipk-1)], i, j, fval, omega);
       }
       else{
         if(ipk != 0){
           bad++;
           if(bad<10){
             printf("Found %d in your blob image at i=%d, j=%d\n",ipk,i,j);}
         }
       }
     } /* j */
   } /* i */
   if(verbose){
     printf("\nFound %d bad pixels in the blob image\n",bad);
   }
}



static char bloboverlaps_doc[] =\
"   success = bloboverlaps (   Numeric.array(blob1, 2D, Int), n1  , res1 \n"\
"                              Numeric.array(blob2, 2D, Int), n2  , res2,\n"\
"                               verbose=0) \n"\
" \n"\
" merges the results from blob1/res1 into blob2/res2\n"\
" blob1 would be the previous image from the series, with its results\n"\
"   if it does not overlap it will stay untouched in res\n"\
"   if it overlaps with next image it is passed into that ones res\n";

int bloboverlaps( INTEGER* b1, INTEGER n1, double* res1, 
                   INTEGER* b2, INTEGER n2, double* res2, 
                   int verbose, int ns, int nf){

    int i, j, safelyneed, ipx ;
    INTEGER *link, p1, p2, ipk, jpk, npk, *T;

  /* Initialise a disjoint set in link 
   * image 2 has peak[i]=i ; i=1->n2
   * image 1 has peak[i]=i+n1+1 ; i=1->n1
   *                          0, 1, 2, 3, n2
   *                          4, 5, 6, 7, 8, 9, n1   need 0, 4 == n1+n2+3
   * link to hold 0->n1-1 ; n1->n2+n2-1 
   * */


  /* This is a disjoint merge operation ... */
  if(verbose) printf("Enter bloboverlaps\n");
  safelyneed=n1+n2+3;
  link =  (int *)malloc(safelyneed*sizeof(INTEGER));
  link[0]=safelyneed;  
  for( i=1 ; i<safelyneed; i++){    
      link[i]=i;
  } 
  /* flag the start of image number 2 */
  link[n2+1]=-99999; /* ==n2=0 Should never be touched by anyone */
  /* results lists of pairs of number */
  /* link holds a disjoint set, we label directly overlapping pixels (i==j) */
  for( i = 0 ; i < ns ; i++ ){   
    for( j = 0 ; j < nf ; j++ ){
        ipx = i*nf+j;
        if( (p1=b1[ipx]) == 0 ) continue;
        if( (p2=b2[ipx]) == 0 ) continue;
        if( link[p2] < 0 || link[p1+n2+1] < 0){
            printf("Whoops!!\n");
            return 0;
        }
        dset_makeunion( link, p2, p1+n2+1 );
    }
  }
  if(verbose) printf("Scanning images\n");
  /* Now we re-label and merge peaks scanning disjoint set */
  for(i=1; i<safelyneed; i++){
    if( link[i] != i && i != n2+1  ){
        j=dset_find( i, link );
        if((i > n2+1) && (j < n2+1)){ /* linking between images */
            jpk = j-1;
            ipk = i-n2-2;
            assert( (n2 > jpk) && (jpk >= 0 ) && ( n1 > ipk) && ( ipk >= 0 ) );
            merge( &res2[NPROPERTY*jpk], &res1[NPROPERTY*ipk]);
            continue;
        }
        if((i > n2+1) && (j > n2+1) ){ /* linking on the same image (1) */
            jpk = j-n2-2;
            ipk = i-n2-2;
            assert( (n2 > jpk) && (jpk >= 0 ) && ( n1 > ipk) && ( ipk >= 0 ) );
            merge( &res1[NPROPERTY*jpk], &res1[NPROPERTY*ipk]);
            continue;
        }
        if( i < n2+1 && j < n2+1 ){ /* linking on the same image (2) */
            jpk = j-1;
            ipk = i-1;
            assert( (n2 > jpk) && (jpk >= 0 ) && ( n1 > ipk) && ( ipk >= 0 ) );
            merge( &res2[NPROPERTY*jpk], &res2[NPROPERTY*ipk]);
            continue;
        }
        assert("I am not here!");
    }
  }


 /* This is the case where two spots on the current image become linked by 
  * by a spot overlap on the previous one
  *
  * The labels are now wrong, in fact there is a single peak in 3D and
  * two peaks in 2D
  *
  * Thanks to Stine West from Riso for finding this subtle bug
  */
    
  /* First, work out the new labels */
  /* Make each T[i] contain the unique ascending integer for the set */ 
  if(verbose) printf("Compress set\n");
  T = (int *) ( malloc( (n2+3)*sizeof(INTEGER) ) );
  assert( T != NULL );

  npk = 0;
  for( i=1 ; i<n2+1; i++){
      if (link[i] == i){
          npk = npk+1;
          T[i] = npk;
      } else {
          j = dset_find(i, link);
          assert( j < i );
          T[i] = T[j];
      }
  }
  if( verbose ){
     for(i=0;i<10;i++) printf("T[%d]=%d ",i,T[i]);
  }
  /* T is now compressed, merge the peaks */
  if(verbose) printf("Merge peaks in res2\n");
  for( i=1; i<n2+1; i++){
      /* dest  = T[i]; */
      /* src   = link[i]; */
      if (link[i] == T[i]) 
          continue;
      if (T[i] < link[i]) { /* copy and zero out */
          if (link[i] == i){ /* This is the place accumulating */
              for (j=0 ; j < NPROPERTY ; j++ ){
                  res2[NPROPERTY*(T[i]-1)+j] = res2[NPROPERTY*(link[i]-1)+j];
                  res2[NPROPERTY*(link[i]-1)+j] = 0;
              }
          } else {
              /* assert this is empty */
              assert( res2[NPROPERTY*(i-1)+s_1] < 0.01 );
          }
          if(verbose>1){
              printf("np i %d j %d %f \n",T[i],link[i],res2[NPROPERTY*T[i]+1]);
          }
      } else {
          assert("Bad logic in bloboverlaps");         
      }
  }
  if(verbose) printf("Relabel image of blobs\n");
  /* Relabel the image where they change */
  for( i = 0 ; i < ns ; i++ ){
      for( j = 0 ; j < nf ; j++ ){
          ipx = i*nf+j;
          if( (p2=b2[ipx]) == 0 ) continue;
          ipk = T[p2];
          if (ipk != p2){
              assert( ipk > 0 && ipk <= n2);
              b2[ipx] = ipk;            
          }
      }
  }
  if(verbose) printf("Done relabelling\n");
  free(T);
  free(link);
  return npk;
}







static char blob_moments_doc [] =                        \
  "   None = blob_moments(Numeric.array(peaks1))\n"        \
  "   \n"                                                \
  "   Loop over array filling out moments from sums\n";

void blob_moments( double * res, int np ){
   compute_moments( res, np );
}



