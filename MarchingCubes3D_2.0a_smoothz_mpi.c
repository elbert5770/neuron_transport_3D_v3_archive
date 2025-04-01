//Typically compiled as unit for a larger project
//Alternatively, use `-DMC-SELF-TEST` to create stand alone executable:
// gcc -DMC_SELF_TEST MarchingCubes.c -o mctest; ./mctest

//------------------------------------------------
// MarchingCubes
//------------------------------------------------
//
// MarchingCubes Algorithm
// Version 0.2 - 12/08/2002
//
// Thomas Lewiner thomas.lewiner@polytechnique.org
// Math Dept, PUC-Rio
//
//
// Translated to C by Ziad S. Saad November 30/04
//________________________________________________
//
// Code downloaded from:
// https://github.com/neurolabusc/nii2mesh
// (A repository owned by the Rorden lab)
// The downloaded code was
// modified for MPI parallelization
// and processing of multiple cells in
// TEM segmentations, e.g. MicronsExplorer database
// by Donald L. Elbert, Department of Neurology
// University of Washington
// Release date: August 7, 2024
// 
// The Rorden lab's interpretation of the absence of a copyright or license is as follows:
// "Don, All the AFNI code from that period is public domain, Lewiner did not include a license. Feel free to share as you wish.
// -c Chris Rorden, PhD Endowed Chair of Neuroimaging Co-Director, McCausland Center for Brain Imaging Department of Psychology
// University of South Carolina"
//
// Modifications to this code by Donald L. Elbert are also released without copyright or license

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <errno.h>
#include <mpi.h>
#include "LookUpTable.h"
#include "MarchingCubes.h"
// #ifdef MC_SELF_TEST
//  #include <unistd.h>
// #endif
#include <unistd.h>
#include <string.h>
// step size of the arrays of vertices and triangles
#define ALLOC_SIZE 16777216
#define MAX_LINE_LENGTH 65536
#define MAX_BUF 1024
#define JOB_TAG 1
#define RESULT_TAG 2
static int debug;
void set_suma_debug(int dbg)
{
   debug = dbg;
   return;
}
//_____________________________________________________________________________
// print cube for debug
void print_cube(MCB *mcb) { printf( "\t%f %f %f %f %f %f %f %f\n", mcb->cube[0], mcb->cube[1], mcb->cube[2], mcb->cube[3], mcb->cube[4], mcb->cube[5], mcb->cube[6], mcb->cube[7]) ; }
//_____________________________________________________________________________

void set_resolution( MCB *mcb, int size_x,  int size_y,  int size_z ) 
{ 
   mcb->size_x = size_x ;  mcb->size_y = size_y ;  mcb->size_z = size_z ; 
   return;
}
void set_method    ( MCB *mcb, int originalMC ) {
    /* originalMC = false is the default */ 
    mcb->originalMC = originalMC ; 
    return;
}

  // Data access
float get_data  (  MCB *mcb, long i,  long j,  long k )  { 
   return (mcb->data[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y]) ; 
}
void  set_data  (  MCB *mcb, float val,  long i,  long j,  long k ) {
  (mcb->data[i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] = val) ; 
}
long   get_x_vert(  MCB *mcb , long i,  long j,  long k )  { return (mcb->x_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] ); }
long   get_y_vert(  MCB *mcb , long i,  long j,  long k )  { return (mcb->y_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] ); }
long   get_z_vert(  MCB *mcb , long i,  long j,  long k )  { return (mcb->z_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] ); }
void  set_x_vert(  MCB *mcb , long val,  long i,  long j,  long k ) { (mcb->x_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] = val ); }
void  set_y_vert(  MCB *mcb , long val,  long i,  long j,  long k ) { (mcb->y_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] = val ); }
void  set_z_vert(  MCB *mcb , long val,  long i,  long j,  long k ) { (mcb->z_verts[ i + j*mcb->size_x + k*mcb->size_x*mcb->size_y] = val ); }







//_____________________________________________________________________________
// Constructor
MCB *MarchingCubes( int size_x, int size_y , int size_z  )
{
// defaults are -1 for all size_ -----------------------------------------------------------------------------
  MCB *mcb=NULL;
  mcb = (MCB *)malloc(sizeof(MCB));
  mcb->originalMC = false;
  mcb->size_x = size_x;
  mcb->size_y = size_y;
  mcb->size_z = size_z;
  mcb->data    =  (float*)NULL;
  mcb->x_verts = ( long *)NULL;
  mcb->y_verts = ( long *)NULL;
  mcb->z_verts =  ( long *)NULL;
  mcb->nverts  =  0;
  mcb->ntrigs  =  0;
  mcb->Nverts  =  0;
  mcb->Ntrigs  =  0;
  mcb->vertices = ( Vertex *)NULL;
  mcb->triangles =(Triangle*)NULL;
  mcb->_case = 0;             /* Was uninitialized and causing weird crashes on linux! ZSS: Oct 06 */
  return(mcb);
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Destructor
void FreeMarchingCubes(MCB *mcb)
//-----------------------------------------------------------------------------
{
  clean_all(mcb) ;
  return;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// main algorithm
void run(MCB *mcb)
//-----------------------------------------------------------------------------
{
   int p;
  //  if (debug) printf("Marching Cubes begin: cpu %ld\n", clock() ) ;

   compute_intersection_points( mcb) ;

   for( mcb->k = 0 ; mcb->k < mcb->size_z-1 ; mcb->k++ )
   for( mcb->j = 0 ; mcb->j < mcb->size_y-1 ; mcb->j++ )
   for( mcb->i = 0 ; mcb->i < mcb->size_x-1 ; mcb->i++ )
   {
    mcb->lut_entry = 0 ;
    for(  p = 0 ; p < 8 ; ++p )
    {
      mcb->cube[p] = get_data( mcb, mcb->i+((p^(p>>1))&1), mcb->j+((p>>1)&1), mcb->k+((p>>2)&1) ) ;
      if( fabs( mcb->cube[p] ) < FLT_EPSILON ) mcb->cube[p] = FLT_EPSILON ;
      if( mcb->cube[p] > 0 ) mcb->lut_entry += 1 << p ;
    }
   /*
    if( ( mcb->cube[0] = get_data( mcb, mcb->i , mcb->j , mcb->k ) ) > 0 ) mcb->lut_entry +=   1 ;
    if( ( mcb->cube[1] = get_data(mcb, mcb->i+1, mcb->j , mcb->k ) ) > 0 ) mcb->lut_entry +=   2 ;
    if( ( mcb->cube[2] = get_data(mcb, mcb->i+1,mcb->j+1, mcb->k ) ) > 0 ) mcb->lut_entry +=   4 ;
    if( ( mcb->cube[3] = get_data(mcb,  mcb->i ,mcb->j+1, mcb->k ) ) > 0 ) mcb->lut_entry +=   8 ;
    if( ( mcb->cube[4] = get_data(mcb,  mcb->i , mcb->j ,mcb->k+1) ) > 0 ) mcb->lut_entry +=  16 ;
    if( ( mcb->cube[5] = get_data(mcb, mcb->i+1, mcb->j ,mcb->k+1) ) > 0 ) mcb->lut_entry +=  32 ;
    if( ( mcb->cube[6] = get_data(mcb, mcb->i+1,mcb->j+1,mcb->k+1) ) > 0 ) mcb->lut_entry +=  64 ;
    if( ( mcb->cube[7] = get_data(mcb,  mcb->i ,mcb->j+1,mcb->k+1) ) > 0 ) mcb->lut_entry += 128 ;
   */
    process_cube( mcb) ;
   }

   if (debug) { 
      printf("Marching Cubes end: cpu %ld\n", clock() ) ;
      for( mcb->i = 0 ; mcb->i < 15 ; mcb->i++ )
      {
       printf("  %7d cases %d\n", mcb->N[mcb->i], mcb->i ) ;
      }
   }
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// init temporary structures (must set sizes before call)
void init_temps(MCB *mcb)
//-----------------------------------------------------------------------------
{
  mcb->data = (float*)calloc((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z, sizeof(float));
  if (mcb->data == NULL) {
      fprintf(stderr, "Failed to allocate memory for mcb->data.\n");
      exit(1);
  }

  mcb->x_verts = (long*)calloc((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z, sizeof(long));
  if (mcb->x_verts == NULL) {
      fprintf(stderr, "Failed to allocate memory for mcb->x_verts.\n");
      exit(1);
  }

  mcb->y_verts = (long*)calloc((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z, sizeof(long));
  if (mcb->y_verts == NULL) {
      fprintf(stderr, "Failed to allocate memory for mcb->y_verts.\n");
      exit(1);
  }

  mcb->z_verts = (long*)calloc((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z, sizeof(long));
  if (mcb->z_verts == NULL) {
      fprintf(stderr, "Failed to allocate memory for mcb->z_verts.\n");
      exit(1);
  }
 
  // printf("verts: %ld %ld %ld\n", mcb->x_verts, mcb->y_verts, mcb->z_verts);
 
  if (mcb->x_verts != NULL) {
    memset( mcb->x_verts, -1, (size_t)((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z * sizeof( long ) )) ;
  } else {
    fprintf(stderr, "mcb->x_verts is NULL.\n");
    exit(1);
  }
  if (mcb->y_verts != NULL) {
    memset( mcb->y_verts, -1, (size_t)((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z * sizeof( long )) ) ;
    } else {
      fprintf(stderr, "mcb->y_verts is NULL.\n");
      exit(1);
  }

  if (mcb->z_verts != NULL) {
  memset( mcb->z_verts, -1, (size_t)((size_t)mcb->size_x * (size_t)mcb->size_y * (size_t)mcb->size_z * sizeof( long )) ) ;
  } else {
      fprintf(stderr, "mcb->z_verts is NULL.\n");
      exit(1);
  }
  if (mcb->N != NULL) {
    memset( mcb->N, 0, 15 * sizeof(int) ) ;
    } else {
      fprintf(stderr, "mcb->N is NULL.\n");
      exit(1);
  }

}
//_____________________________________________________________________________



//_____________________________________________________________________________
// init all structures (must set sizes before call)
void init_all (MCB *mcb)
//-----------------------------------------------------------------------------
{
  init_temps(mcb) ;

  mcb->nverts = mcb->ntrigs = 0 ;
  mcb->Nverts = mcb->Ntrigs = ALLOC_SIZE ;
  mcb->vertices  = (Vertex*)calloc(mcb->Nverts, sizeof(Vertex)) ;
  mcb->triangles = (Triangle*)calloc(mcb->Ntrigs, sizeof(Triangle));
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// clean temporary structures
void clean_temps(MCB *mcb)
//-----------------------------------------------------------------------------
{
  free(mcb->data); 
  free(mcb->x_verts);
  free(mcb->y_verts);
  free(mcb->z_verts);

  mcb->data     = (float*)NULL ;
  mcb->x_verts  = (long*)NULL ;
  mcb->y_verts  = (long*)NULL ;
  mcb->z_verts  = (long*)NULL ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// clean all structures
void clean_all(MCB *mcb)
//-----------------------------------------------------------------------------
{
  clean_temps(mcb) ;
  free(mcb->vertices)  ;
  free(mcb->triangles) ;
  mcb->vertices  = (Vertex   *)NULL ;
  mcb->triangles = (Triangle *)NULL ;
  mcb->nverts = mcb->ntrigs = 0 ;
  mcb->Nverts = mcb->Ntrigs = 0 ;

  mcb->size_x = mcb->size_y = mcb->size_z = -1 ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
//_____________________________________________________________________________


//_____________________________________________________________________________
// Compute the intersection points
void compute_intersection_points(MCB *mcb )
//-----------------------------------------------------------------------------
{
  for( mcb->k = 0 ; mcb->k < mcb->size_z ; mcb->k++ )
  for( mcb->j = 0 ; mcb->j < mcb->size_y ; mcb->j++ )
  for( mcb->i = 0 ; mcb->i < mcb->size_x ; mcb->i++ )
  {
    mcb->cube[0] = get_data( mcb, mcb->i, mcb->j, mcb->k ) ;
    if( mcb->i < mcb->size_x - 1 ) mcb->cube[1] = get_data(mcb, mcb->i+1, mcb->j , mcb->k ) ;
    else                 mcb->cube[1] = mcb->cube[0] ;

    if( mcb->j < mcb->size_y - 1 ) mcb->cube[3] = get_data( mcb, mcb->i ,mcb->j+1, mcb->k ) ;
    else                 mcb->cube[3] = mcb->cube[0] ;

    if( mcb->k < mcb->size_z - 1 ) mcb->cube[4] = get_data( mcb, mcb->i , mcb->j ,mcb->k+1) ;
    else                 mcb->cube[4] = mcb->cube[0] ;

    if( fabs( mcb->cube[0] ) < FLT_EPSILON ) mcb->cube[0] = FLT_EPSILON ;
    if( fabs( mcb->cube[1] ) < FLT_EPSILON ) mcb->cube[1] = FLT_EPSILON ;
    if( fabs( mcb->cube[3] ) < FLT_EPSILON ) mcb->cube[3] = FLT_EPSILON ;
    if( fabs( mcb->cube[4] ) < FLT_EPSILON ) mcb->cube[4] = FLT_EPSILON ;

    if( mcb->cube[0] < 0 )
    {
      if( mcb->cube[1] > 0 ) set_x_vert( mcb, add_x_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
      if( mcb->cube[3] > 0 ) set_y_vert( mcb, add_y_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
      if( mcb->cube[4] > 0 ) set_z_vert( mcb, add_z_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
    }
    else
    {
      if( mcb->cube[1] < 0 ) set_x_vert( mcb, add_x_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
      if( mcb->cube[3] < 0 ) set_y_vert( mcb, add_y_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
      if( mcb->cube[4] < 0 ) set_z_vert( mcb, add_z_vertex( mcb), mcb->i,mcb->j,mcb->k ) ;
    }
  }
}
//_____________________________________________________________________________

//_____________________________________________________________________________
// Test a face
// if face>0 return true if the face contains a part of the surface
int test_face( MCB *mcb, schar face )
//-----------------------------------------------------------------------------
{
  float A,B,C,D ;

  switch( face )
  {
  case -1 : case 1 :  A = mcb->cube[0] ;  B = mcb->cube[4] ;  C = mcb->cube[5] ;  D = mcb->cube[1] ;  break ;
  case -2 : case 2 :  A = mcb->cube[1] ;  B = mcb->cube[5] ;  C = mcb->cube[6] ;  D = mcb->cube[2] ;  break ;
  case -3 : case 3 :  A = mcb->cube[2] ;  B = mcb->cube[6] ;  C = mcb->cube[7] ;  D = mcb->cube[3] ;  break ;
  case -4 : case 4 :  A = mcb->cube[3] ;  B = mcb->cube[7] ;  C = mcb->cube[4] ;  D = mcb->cube[0] ;  break ;
  case -5 : case 5 :  A = mcb->cube[0] ;  B = mcb->cube[3] ;  C = mcb->cube[2] ;  D = mcb->cube[1] ;  break ;
  case -6 : case 6 :  A = mcb->cube[4] ;  B = mcb->cube[7] ;  C = mcb->cube[6] ;  D = mcb->cube[5] ;  break ;
  default : printf( "Invalid face code %d %d %d: %d\n",  mcb->i,  mcb->j,  mcb->k, face ) ;  print_cube(mcb) ;  A = B = C = D = 0 ;
  };

  if( fabs( A*C - B*D ) < FLT_EPSILON )
    return face >= 0 ;
  return face * A * ( A*C - B*D ) >= 0  ;  // face and A invert signs
}
/*
{
  float A,B,C,D ;

  switch( face )
  {
  case -1 : case 1 :  A = mcb->cube[0] ;  B = mcb->cube[4] ;  C = mcb->cube[5] ;  D = mcb->cube[1] ;  break ;
  case -2 : case 2 :  A = mcb->cube[1] ;  B = mcb->cube[5] ;  C = mcb->cube[6] ;  D = mcb->cube[2] ;  break ;
  case -3 : case 3 :  A = mcb->cube[2] ;  B = mcb->cube[6] ;  C = mcb->cube[7] ;  D = mcb->cube[3] ;  break ;
  case -4 : case 4 :  A = mcb->cube[3] ;  B = mcb->cube[7] ;  C = mcb->cube[4] ;  D = mcb->cube[0] ;  break ;
  case -5 : case 5 :  A = mcb->cube[0] ;  B = mcb->cube[3] ;  C = mcb->cube[2] ;  D = mcb->cube[1] ;  break ;
  case -6 : case 6 :  A = mcb->cube[4] ;  B = mcb->cube[7] ;  C = mcb->cube[6] ;  D = mcb->cube[5] ;  break ;
  default : printf( "Invalid face code %d\n", face ) ;  print_cube(mcb) ;  A = B = C = D = 0 ;
  };

  return (face * A * ( A*C - B*D ) >= 0)  ;  // face and A invert signs
}*/
//_____________________________________________________________________________





//_____________________________________________________________________________
// Test the interior of a cube
// if s == 7, return true  if the interior is empty
// if s ==-7, return false if the interior is empty
int test_interior( MCB *mcb, schar s )
//-----------------------------------------------------------------------------
{
  float t, At=0, Bt=0, Ct=0, Dt=0, a, b ;
  char  test =  0 ;
  char  edge = -1 ; // reference edge of the triangulation

  switch( mcb->_case )
  {
  case  4 :
  case 10 :
    a = ( mcb->cube[4] - mcb->cube[0] ) * ( mcb->cube[6] - mcb->cube[2] ) - ( mcb->cube[7] - mcb->cube[3] ) * ( mcb->cube[5] - mcb->cube[1] ) ;
    b =  mcb->cube[2] * ( mcb->cube[4] - mcb->cube[0] ) + mcb->cube[0] * ( mcb->cube[6] - mcb->cube[2] )
             - mcb->cube[1] * ( mcb->cube[7] - mcb->cube[3] ) - mcb->cube[3] * ( mcb->cube[5] - mcb->cube[1] ) ;
    t = - b / (2*a) ;
    if( t<0 || t>1 ) return s>0 ;

    At = mcb->cube[0] + ( mcb->cube[4] - mcb->cube[0] ) * t ;
    Bt = mcb->cube[3] + ( mcb->cube[7] - mcb->cube[3] ) * t ;
    Ct = mcb->cube[2] + ( mcb->cube[6] - mcb->cube[2] ) * t ;
    Dt = mcb->cube[1] + ( mcb->cube[5] - mcb->cube[1] ) * t ;
    break ;

  case  6 :
  case  7 :
  case 12 :
  case 13 :
    switch( mcb->_case )
    {
    case  6 : edge = test6 [mcb->config][2] ; break ;
    case  7 : edge = test7 [mcb->config][4] ; break ;
    case 12 : edge = test12[mcb->config][3] ; break ;
    case 13 : edge = tiling13_5_1[mcb->config][mcb->subconfig][0] ; break ;
    }
    switch( edge )
    {
    case  0 :
      t  = mcb->cube[0] / ( mcb->cube[0] - mcb->cube[1] ) ;
      At = 0 ;
      Bt = mcb->cube[3] + ( mcb->cube[2] - mcb->cube[3] ) * t ;
      Ct = mcb->cube[7] + ( mcb->cube[6] - mcb->cube[7] ) * t ;
      Dt = mcb->cube[4] + ( mcb->cube[5] - mcb->cube[4] ) * t ;
      break ;
    case  1 :
      t  = mcb->cube[1] / ( mcb->cube[1] - mcb->cube[2] ) ;
      At = 0 ;
      Bt = mcb->cube[0] + ( mcb->cube[3] - mcb->cube[0] ) * t ;
      Ct = mcb->cube[4] + ( mcb->cube[7] - mcb->cube[4] ) * t ;
      Dt = mcb->cube[5] + ( mcb->cube[6] - mcb->cube[5] ) * t ;
      break ;
    case  2 :
      t  = mcb->cube[2] / ( mcb->cube[2] - mcb->cube[3] ) ;
      At = 0 ;
      Bt = mcb->cube[1] + ( mcb->cube[0] - mcb->cube[1] ) * t ;
      Ct = mcb->cube[5] + ( mcb->cube[4] - mcb->cube[5] ) * t ;
      Dt = mcb->cube[6] + ( mcb->cube[7] - mcb->cube[6] ) * t ;
      break ;
    case  3 :
      t  = mcb->cube[3] / ( mcb->cube[3] - mcb->cube[0] ) ;
      At = 0 ;
      Bt = mcb->cube[2] + ( mcb->cube[1] - mcb->cube[2] ) * t ;
      Ct = mcb->cube[6] + ( mcb->cube[5] - mcb->cube[6] ) * t ;
      Dt = mcb->cube[7] + ( mcb->cube[4] - mcb->cube[7] ) * t ;
      break ;
    case  4 :
      t  = mcb->cube[4] / ( mcb->cube[4] - mcb->cube[5] ) ;
      At = 0 ;
      Bt = mcb->cube[7] + ( mcb->cube[6] - mcb->cube[7] ) * t ;
      Ct = mcb->cube[3] + ( mcb->cube[2] - mcb->cube[3] ) * t ;
      Dt = mcb->cube[0] + ( mcb->cube[1] - mcb->cube[0] ) * t ;
      break ;
    case  5 :
      t  = mcb->cube[5] / ( mcb->cube[5] - mcb->cube[6] ) ;
      At = 0 ;
      Bt = mcb->cube[4] + ( mcb->cube[7] - mcb->cube[4] ) * t ;
      Ct = mcb->cube[0] + ( mcb->cube[3] - mcb->cube[0] ) * t ;
      Dt = mcb->cube[1] + ( mcb->cube[2] - mcb->cube[1] ) * t ;
      break ;
    case  6 :
      t  = mcb->cube[6] / ( mcb->cube[6] - mcb->cube[7] ) ;
      At = 0 ;
      Bt = mcb->cube[5] + ( mcb->cube[4] - mcb->cube[5] ) * t ;
      Ct = mcb->cube[1] + ( mcb->cube[0] - mcb->cube[1] ) * t ;
      Dt = mcb->cube[2] + ( mcb->cube[3] - mcb->cube[2] ) * t ;
      break ;
    case  7 :
      t  = mcb->cube[7] / ( mcb->cube[7] - mcb->cube[4] ) ;
      At = 0 ;
      Bt = mcb->cube[6] + ( mcb->cube[5] - mcb->cube[6] ) * t ;
      Ct = mcb->cube[2] + ( mcb->cube[1] - mcb->cube[2] ) * t ;
      Dt = mcb->cube[3] + ( mcb->cube[0] - mcb->cube[3] ) * t ;
      break ;
    case  8 :
      t  = mcb->cube[0] / ( mcb->cube[0] - mcb->cube[4] ) ;
      At = 0 ;
      Bt = mcb->cube[3] + ( mcb->cube[7] - mcb->cube[3] ) * t ;
      Ct = mcb->cube[2] + ( mcb->cube[6] - mcb->cube[2] ) * t ;
      Dt = mcb->cube[1] + ( mcb->cube[5] - mcb->cube[1] ) * t ;
      break ;
    case  9 :
      t  = mcb->cube[1] / ( mcb->cube[1] - mcb->cube[5] ) ;
      At = 0 ;
      Bt = mcb->cube[0] + ( mcb->cube[4] - mcb->cube[0] ) * t ;
      Ct = mcb->cube[3] + ( mcb->cube[7] - mcb->cube[3] ) * t ;
      Dt = mcb->cube[2] + ( mcb->cube[6] - mcb->cube[2] ) * t ;
      break ;
    case 10 :
      t  = mcb->cube[2] / ( mcb->cube[2] - mcb->cube[6] ) ;
      At = 0 ;
      Bt = mcb->cube[1] + ( mcb->cube[5] - mcb->cube[1] ) * t ;
      Ct = mcb->cube[0] + ( mcb->cube[4] - mcb->cube[0] ) * t ;
      Dt = mcb->cube[3] + ( mcb->cube[7] - mcb->cube[3] ) * t ;
      break ;
    case 11 :
      t  = mcb->cube[3] / ( mcb->cube[3] - mcb->cube[7] ) ;
      At = 0 ;
      Bt = mcb->cube[2] + ( mcb->cube[6] - mcb->cube[2] ) * t ;
      Ct = mcb->cube[1] + ( mcb->cube[5] - mcb->cube[1] ) * t ;
      Dt = mcb->cube[0] + ( mcb->cube[4] - mcb->cube[0] ) * t ;
      break ;
    default : printf( "Invalid edge %d\n", edge ) ;  print_cube(mcb) ;  break ;
    }
    break ;

  default : printf( "Invalid ambiguous case %d\n", mcb->_case ) ;  print_cube(mcb) ;  break ;
  }

  if( At >= 0 ) test ++ ;
  if( Bt >= 0 ) test += 2 ;
  if( Ct >= 0 ) test += 4 ;
  if( Dt >= 0 ) test += 8 ;
  switch( test )
  {
  case  0 : return s>0 ;
  case  1 : return s>0 ;
  case  2 : return s>0 ;
  case  3 : return s>0 ;
  case  4 : return s>0 ;
  case  5 : if( At * Ct - Bt * Dt <  FLT_EPSILON ) return s>0 ; break ;
  case  6 : return s>0 ;
  case  7 : return s<0 ;
  case  8 : return s>0 ;
  case  9 : return s>0 ;
  case 10 : if( At * Ct - Bt * Dt >= FLT_EPSILON ) return s>0 ; break ;
  case 11 : return s<0 ;
  case 12 : return s>0 ;
  case 13 : return s<0 ;
  case 14 : return s<0 ;
  case 15 : return s<0 ;
  }

  return s<0 ;
}
//_____________________________________________________________________________

//_____________________________________________________________________________
// Process a unit cube
void process_cube( MCB *mcb)
//-----------------------------------------------------------------------------
{
  int   v12 = -1 ;
  /* print_cube(mcb) ; 
  fprintf (stderr,"_case=%d\n", mcb->_case);
  fprintf (stderr,"N=%d\n", mcb->N[mcb->_case]);*/
  if (mcb->_case >= N_MAX) {
   fprintf (stderr,"Unexpected _case value of %d\nResetting to 0.\n",mcb->_case);
   mcb->_case = 0; 
  }
  mcb->N[mcb->_case]++ ;

  if( mcb->originalMC )
  {
    char nt = 0 ;
    while( casesClassic[mcb->lut_entry][3*nt] != -1 ) nt++ ;
    add_triangle(mcb, casesClassic[mcb->lut_entry], nt, -1 ) ;
    return ;
  }

  mcb->_case   = cases[mcb->lut_entry][0] ;
  mcb->config = cases[mcb->lut_entry][1] ;
  mcb->subconfig = 0 ;

  switch( mcb->_case )
  {
  case  0 :
    break ;

  case  1 :
    add_triangle(mcb, tiling1[mcb->config], 1, -1) ;
    break ;

  case  2 :
    add_triangle(mcb, tiling2[mcb->config], 2, -1) ;
    break ;

  case  3 :
    if( test_face(mcb, test3[mcb->config]) )
      add_triangle(mcb,  tiling3_2[mcb->config], 4, -1) ; // 3.2
    else
      add_triangle(mcb,  tiling3_1[mcb->config], 2, -1) ; // 3.1
    break ;

  case  4 :
    if( test_interior(mcb, test4[mcb->config]) )
      add_triangle(mcb, tiling4_1[mcb->config], 2, -1) ; // 4.1.1
    else
      add_triangle(mcb, tiling4_2[mcb->config], 6, -1) ; // 4.1.2
    break ;

  case  5 :
    add_triangle(mcb, tiling5[mcb->config], 3, -1) ;
    break ;

  case  6 :
    if( test_face(mcb, test6[mcb->config][0]) )
      add_triangle(mcb, tiling6_2[mcb->config], 5, -1) ; // 6.2
    else
    {
      if( test_interior(mcb, test6[mcb->config][1]) )
        add_triangle(mcb, tiling6_1_1[mcb->config], 3, -1) ; // 6.1.1
      else
    {
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling6_1_2[mcb->config], 9 , v12) ; // 6.1.2
      }
    }
    break ;
  case  7 :
    if( test_face(mcb, test7[mcb->config][0] ) ) mcb->subconfig +=  1 ;
    if( test_face(mcb, test7[mcb->config][1] ) ) mcb->subconfig +=  2 ;
    if( test_face(mcb, test7[mcb->config][2] ) ) mcb->subconfig +=  4 ;
    switch( mcb->subconfig )
      {
      case 0 :
        add_triangle(mcb, tiling7_1[mcb->config], 3, -1) ; break ;
      case 1 :
        add_triangle(mcb, tiling7_2[mcb->config][0], 5, -1) ; break ;
      case 2 :
        add_triangle(mcb, tiling7_2[mcb->config][1], 5, -1) ; break ;
      case 3 :
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling7_3[mcb->config][0], 9, v12 ) ; break ;
      case 4 :
        add_triangle(mcb, tiling7_2[mcb->config][2], 5, -1) ; break ;
      case 5 :
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling7_3[mcb->config][1], 9, v12 ) ; break ;
      case 6 :
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling7_3[mcb->config][2], 9, v12 ) ; break ;
      case 7 :
        if( test_interior(mcb, test7[mcb->config][3]) )
          add_triangle(mcb, tiling7_4_2[mcb->config], 9, -1) ;
        else
          add_triangle(mcb, tiling7_4_1[mcb->config], 5, -1) ;
        break ;
      };
    break ;

  case  8 :
    add_triangle(mcb, tiling8[mcb->config], 2, -1) ;
    break ;

  case  9 :
    add_triangle(mcb, tiling9[mcb->config], 4, -1) ;
    break ;

  case 10 :
    if( test_face(mcb, test10[mcb->config][0]) )
    {
      if( test_face(mcb, test10[mcb->config][1]) )
        add_triangle(mcb, tiling10_1_1_[mcb->config], 4, -1) ; // 10.1.1
      else
      {
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling10_2[mcb->config], 8, v12 ) ; // 10.2
      }
    }
    else
    {
      if( test_face(mcb, test10[mcb->config][1]) )
      {
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling10_2_[mcb->config], 8, v12 ) ; // 10.2
      }
      else
      {
        if( test_interior(mcb, test10[mcb->config][2]) )
          add_triangle(mcb, tiling10_1_1[mcb->config], 4, -1) ; // 10.1.1
        else
          add_triangle(mcb, tiling10_1_2[mcb->config], 8, -1) ; // 10.1.2
      }
    }
    break ;

  case 11 :
    add_triangle(mcb, tiling11[mcb->config], 4, -1) ;
    break ;

  case 12 :
    if( test_face(mcb, test12[mcb->config][0]) )
    {
      if( test_face(mcb, test12[mcb->config][1]) )
        add_triangle(mcb, tiling12_1_1_[mcb->config], 4, -1) ; // 12.1.1
      else
      {
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling12_2[mcb->config], 8, v12 ) ; // 12.2
      }
    }
    else
    {
      if( test_face(mcb, test12[mcb->config][1]) )
      {
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling12_2_[mcb->config], 8, v12 ) ; // 12.2
      }
      else
      {
        if( test_interior(mcb, test12[mcb->config][2]) )
          add_triangle(mcb, tiling12_1_1[mcb->config], 4, -1) ; // 12.1.1
        else
          add_triangle(mcb, tiling12_1_2[mcb->config], 8, -1) ; // 12.1.2
      }
    }
    break ;

  case 13 :
    if( test_face(mcb,  test13[mcb->config][0] ) ) mcb->subconfig +=  1 ;
    if( test_face(mcb,  test13[mcb->config][1] ) ) mcb->subconfig +=  2 ;
    if( test_face(mcb,  test13[mcb->config][2] ) ) mcb->subconfig +=  4 ;
    if( test_face(mcb,  test13[mcb->config][3] ) ) mcb->subconfig +=  8 ;
    if( test_face(mcb,  test13[mcb->config][4] ) ) mcb->subconfig += 16 ;
    if( test_face(mcb,  test13[mcb->config][5] ) ) mcb->subconfig += 32 ;
    switch( subconfig13[mcb->subconfig] )
    {
      case 0 :/* 13.1 */
        add_triangle(mcb,  tiling13_1[mcb->config], 4, -1) ; break ;

      case 1 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][0], 6, -1) ; break ;
      case 2 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][1], 6, -1) ; break ;
      case 3 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][2], 6, -1) ; break ;
      case 4 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][3], 6, -1) ; break ;
      case 5 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][4], 6, -1) ; break ;
      case 6 :/* 13.2 */
        add_triangle(mcb,  tiling13_2[mcb->config][5], 6, -1) ; break ;

      case 7 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][0], 10, v12 ) ; break ;
      case 8 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][1], 10, v12 ) ; break ;
      case 9 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][2], 10, v12 ) ; break ;
      case 10 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][3], 10, v12 ) ; break ;
      case 11 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][4], 10, v12 ) ; break ;
      case 12 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][5], 10, v12 ) ; break ;
      case 13 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][6], 10, v12 ) ; break ;
      case 14 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][7], 10, v12 ) ; break ;
      case 15 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][8], 10, v12 ) ; break ;
      case 16 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][9], 10, v12 ) ; break ;
      case 17 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][10], 10, v12 ) ; break ;
      case 18 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb,  tiling13_3[mcb->config][11], 10, v12 ) ; break ;

      case 19 :/* 13.4 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_4[mcb->config][0], 12, v12 ) ; break ;
      case 20 :/* 13.4 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_4[mcb->config][1], 12, v12 ) ; break ;
      case 21 :/* 13.4 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_4[mcb->config][2], 12, v12 ) ; break ;
      case 22 :/* 13.4 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_4[mcb->config][3], 12, v12 ) ; break ;

      case 23 :/* 13.5 */
        mcb->subconfig = 0 ;
        if( test_interior(mcb, test13[mcb->config][6] ) )
          add_triangle(mcb, tiling13_5_1[mcb->config][0], 6, -1) ;
        else
          add_triangle(mcb, tiling13_5_2[mcb->config][0], 10, -1) ;
        break ;
      case 24 :/* 13.5 */
        mcb->subconfig = 1 ;
        if( test_interior(mcb, test13[mcb->config][6] ) )
          add_triangle(mcb, tiling13_5_1[mcb->config][1], 6, -1) ;
        else
          add_triangle(mcb, tiling13_5_2[mcb->config][1], 10, -1) ;
        break ;
      case 25 :/* 13.5 */
        mcb->subconfig = 2 ;
        if( test_interior(mcb, test13[mcb->config][6] ) )
          add_triangle(mcb, tiling13_5_1[mcb->config][2], 6, -1) ;
        else
          add_triangle(mcb, tiling13_5_2[mcb->config][2], 10, -1) ;
        break ;
      case 26 :/* 13.5 */
        mcb->subconfig = 3 ;
        if( test_interior(mcb, test13[mcb->config][6] ) )
          add_triangle(mcb, tiling13_5_1[mcb->config][3], 6, -1) ;
        else
          add_triangle(mcb, tiling13_5_2[mcb->config][3], 10, -1) ;
        break ;


      case 27 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][0], 10, v12 ) ; break ;
      case 28 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][1], 10, v12 ) ; break ;
      case 29 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][2], 10, v12 ) ; break ;
      case 30 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][3], 10, v12 ) ; break ;
      case 31 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][4], 10, v12 ) ; break ;
      case 32 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][5], 10, v12 ) ; break ;
      case 33 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][6], 10, v12 ) ; break ;
      case 34 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][7], 10, v12 ) ; break ;
      case 35 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][8], 10, v12 ) ; break ;
      case 36 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][9], 10, v12 ) ; break ;
      case 37 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][10], 10, v12 ) ; break ;
      case 38 :/* 13.3 */
        v12 = add_c_vertex(mcb) ;
        add_triangle(mcb, tiling13_3_[mcb->config][11], 10, v12 ) ; break ;

      case 39 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][0], 6, -1) ; break ;
      case 40 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][1], 6, -1) ; break ;
      case 41 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][2], 6, -1) ; break ;
      case 42 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][3], 6, -1) ; break ;
      case 43 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][4], 6, -1) ; break ;
      case 44 :/* 13.2 */
        add_triangle(mcb, tiling13_2_[mcb->config][5], 6, -1) ; break ;

      case 45 :/* 13.1 */
        add_triangle(mcb, tiling13_1_[mcb->config], 4, -1) ; break ;

      default :
        printf("Marching Cubes: Impossible case 13?\n" ) ;  print_cube(mcb) ;
      }
      break ;

  case 14 :
    add_triangle(mcb, tiling14[mcb->config], 4, -1) ;
    break ;
  };
}

//_____________________________________________________________________________



//_____________________________________________________________________________
// Adding triangles
void add_triangle( MCB *mcb , const char* trig, char n, int v12 )
//-----------------------------------------------------------------------------
{
  int   t, tv[3] ;
//printf( "+>> %d %d %d\n", mcb->i  , mcb->j , mcb->k);

  for( t = 0 ; t < 3*n ; t++ )
  {
    switch( trig[t] )
    {
    case  0 : tv[ t % 3 ] = get_x_vert(mcb,  mcb->i  , mcb->j , mcb->k ) ; break ;
    case  1 : tv[ t % 3 ] = get_y_vert(mcb, mcb->i +1, mcb->j , mcb->k ) ; break ;
    case  2 : tv[ t % 3 ] = get_x_vert(mcb,  mcb->i  ,mcb->j+1, mcb->k ) ; break ;
    case  3 : tv[ t % 3 ] = get_y_vert(mcb,  mcb->i  , mcb->j , mcb->k ) ; break ;
    case  4 : tv[ t % 3 ] = get_x_vert(mcb,  mcb->i  , mcb->j ,mcb->k+1) ; break ;
    case  5 : tv[ t % 3 ] = get_y_vert(mcb, mcb->i +1, mcb->j ,mcb->k+1) ; break ;
    case  6 : tv[ t % 3 ] = get_x_vert(mcb,  mcb->i  ,mcb->j+1,mcb->k+1) ; break ;
    case  7 : tv[ t % 3 ] = get_y_vert(mcb,  mcb->i  , mcb->j ,mcb->k+1) ; break ;
    case  8 : tv[ t % 3 ] = get_z_vert(mcb,  mcb->i  , mcb->j , mcb->k ) ; break ;
    case  9 : tv[ t % 3 ] = get_z_vert(mcb, mcb->i +1, mcb->j , mcb->k ) ; break ;
    case 10 : tv[ t % 3 ] = get_z_vert(mcb, mcb->i +1,mcb->j+1, mcb->k ) ; break ;
    case 11 : tv[ t % 3 ] = get_z_vert(mcb,  mcb->i  ,mcb->j+1, mcb->k ) ; break ;
    case 12 : tv[ t % 3 ] = v12 ; break ;
    default : break ;
    }

    if( tv[t%3] == -1 )
    {
      printf("Marching Cubes: invalid triangle %d\n", mcb->ntrigs+1) ;
      print_cube(mcb) ;
    }

    if( t%3 == 2 )
    { 
      Triangle *T = NULL;
      if( mcb->ntrigs >= mcb->Ntrigs )
      {
        Triangle *temp = mcb->triangles ;
        mcb->triangles = (Triangle*)malloc(2*mcb->Ntrigs * sizeof(Triangle));
        memcpy( mcb->triangles, temp, mcb->Ntrigs*sizeof(Triangle) ) ;
        free(temp) ; temp = NULL;
        if (debug) printf("%d allocated triangles\n", mcb->Ntrigs) ;
        mcb->Ntrigs *= 2 ;
      }

      T = mcb->triangles + mcb->ntrigs++ ;
      T->v1    = tv[0] ;
      T->v2    = tv[1] ;
      T->v3    = tv[2] ;
    }
  }
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Calculating gradient

float get_x_grad( MCB *mcb,  long i,  long j,  long k ) 
//-----------------------------------------------------------------------------
{
  if( i > 0 )
  {
    if ( i < mcb->size_x - 1 )
      return (( get_data( mcb, i+1, j, k ) - get_data( mcb, i-1, j, k ) ) / 2) ;
    else
      return (get_data( mcb, i, j, k ) - get_data( mcb, i-1, j, k )) ;
  }
  else
    return (get_data( mcb, i+1, j, k ) - get_data( mcb, i, j, k )) ;
}
//-----------------------------------------------------------------------------

float get_y_grad( MCB *mcb,  long i,  long j,  long k ) 
//-----------------------------------------------------------------------------
{
  if( j > 0 )
  {
    if ( j < mcb->size_y - 1 )
      return (( get_data( mcb, i, j+1, k ) - get_data( mcb, i, j-1, k ) ) / 2) ;
    else
      return (get_data( mcb, i, j, k ) - get_data( mcb, i, j-1, k )) ;
  }
  else
    return (get_data(mcb,  i, j+1, k ) - get_data(mcb, i, j, k )) ;
}
//-----------------------------------------------------------------------------

float get_z_grad( MCB *mcb, long i,  long j,  long k ) 
//-----------------------------------------------------------------------------
{
  if( k > 0 )
  {
    if ( k < mcb->size_z - 1 )
      return (( get_data( mcb, i, j, k+1 ) - get_data( mcb, i, j, k-1 ) ) / 2) ;
    else
      return (get_data( mcb, i, j, k ) - get_data( mcb, i, j, k-1 )) ;
  }
  else
    return (get_data( mcb, i, j, k+1 ) - get_data( mcb, i, j, k )) ;
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// Adding vertices

void test_vertex_addition(MCB *mcb)
{
  if( mcb->nverts >= mcb->Nverts )
  {
    Vertex *temp = mcb->vertices ;
    mcb->vertices =  (Vertex*)malloc(mcb->Nverts*2 * sizeof(Vertex)) ;
    memcpy( mcb->vertices, temp, mcb->Nverts*sizeof(Vertex) ) ;
    free(temp); temp = NULL;
    if (debug) printf("%d allocated vertices\n", mcb->Nverts) ;
    mcb->Nverts *= 2 ;
  }
}


int add_x_vertex(MCB *mcb )
//-----------------------------------------------------------------------------
{
   Vertex *vert;
   float   u;
  
  test_vertex_addition(mcb) ;
  vert = mcb->vertices + mcb->nverts++ ;

  u = ( mcb->cube[0] ) / ( mcb->cube[0] - mcb->cube[1] ) ;

  vert->x      = (float)mcb->i+u;
  vert->y      = (float) mcb->j ;
  vert->z      = (float) mcb->k ;

  vert->nx = (1-u)*get_x_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_x_grad(mcb, mcb->i+1,mcb->j,mcb->k) ;
  vert->ny = (1-u)*get_y_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_y_grad(mcb, mcb->i+1,mcb->j,mcb->k) ;
  vert->nz = (1-u)*get_z_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_z_grad(mcb, mcb->i+1,mcb->j,mcb->k) ;

  u = (float) sqrt( vert->nx * vert->nx + vert->ny * vert->ny +vert->nz * vert->nz ) ;
  
  if( u > 0 )
  {
    vert->nx /= u ;
    vert->ny /= u ;
    vert->nz /= u ;
  }


  return (mcb->nverts-1) ;
}
//-----------------------------------------------------------------------------

int add_y_vertex( MCB *mcb)
//-----------------------------------------------------------------------------
{  Vertex *vert;
   float u;
  test_vertex_addition(mcb) ;
  vert = mcb->vertices + mcb->nverts++ ;

  u = ( mcb->cube[0] ) / ( mcb->cube[0] - mcb->cube[3] ) ;

  vert->x      = (float) mcb->i ;
  vert->y      = (float)mcb->j+u;
  vert->z      = (float) mcb->k ;

  vert->nx = (1-u)*get_x_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_x_grad(mcb, mcb->i,mcb->j+1,mcb->k) ;
  vert->ny = (1-u)*get_y_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_y_grad(mcb, mcb->i,mcb->j+1,mcb->k) ;
  vert->nz = (1-u)*get_z_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_z_grad(mcb, mcb->i,mcb->j+1,mcb->k) ;

  u = (float) sqrt( vert->nx * vert->nx + vert->ny * vert->ny +vert->nz * vert->nz ) ;
  if( u > 0 )
  {
    vert->nx /= u ;
    vert->ny /= u ;
    vert->nz /= u ;
  }

  return (mcb->nverts-1) ;
}
//-----------------------------------------------------------------------------

int add_z_vertex(MCB *mcb )
//-----------------------------------------------------------------------------
{  Vertex *vert;
   float u;
  test_vertex_addition(mcb) ;
  vert = mcb->vertices + mcb->nverts++ ;

  u = ( mcb->cube[0] ) / ( mcb->cube[0] - mcb->cube[4] ) ;

  vert->x      = (float) mcb->i ;
  vert->y      = (float) mcb->j ;
  vert->z      = (float)mcb->k+u;

  vert->nx = (1-u)*get_x_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_x_grad(mcb, mcb->i,mcb->j,mcb->k+1) ;
  vert->ny = (1-u)*get_y_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_y_grad(mcb, mcb->i,mcb->j,mcb->k+1) ;
  vert->nz = (1-u)*get_z_grad(mcb, mcb->i,mcb->j,mcb->k) + u*get_z_grad(mcb, mcb->i,mcb->j,mcb->k+1) ;

  u = (float) sqrt( vert->nx * vert->nx + vert->ny * vert->ny +vert->nz * vert->nz ) ;
  if( u > 0 )
  {
    vert->nx /= u ;
    vert->ny /= u ;
    vert->nz /= u ;
  }

  return (mcb->nverts-1) ;
}


int add_c_vertex( MCB *mcb)
//-----------------------------------------------------------------------------
{  Vertex *vert, v;
   float u;
   int   vid ;
  test_vertex_addition(mcb) ;
  vert = mcb->vertices + mcb->nverts++ ;

  u = 0 ;

  vert->x = vert->y = vert->z =  vert->nx = vert->ny = vert->nz = 0 ;

  // Computes the average of the intersection points of the cube
  vid = get_x_vert( mcb, mcb->i , mcb->j , mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_y_vert(mcb, mcb->i+1, mcb->j , mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_x_vert( mcb, mcb->i ,mcb->j+1, mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_y_vert( mcb, mcb->i , mcb->j , mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_x_vert( mcb, mcb->i , mcb->j ,mcb->k+1) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_y_vert(mcb, mcb->i+1, mcb->j ,mcb->k+1) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_x_vert( mcb, mcb->i ,mcb->j+1,mcb->k+1) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_y_vert(mcb,  mcb->i , mcb->j ,mcb->k+1) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_z_vert( mcb, mcb->i , mcb->j , mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_z_vert(mcb, mcb->i+1, mcb->j , mcb->k ) ;
  if( vid != -1 ) { ++u ;  v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_z_vert(mcb, mcb->i+1,mcb->j+1, mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }
  vid = get_z_vert( mcb, mcb->i ,mcb->j+1, mcb->k ) ;
  if( vid != -1 ) { ++u ;   v = mcb->vertices[vid] ; vert->x += v.x ;  vert->y += v.y ;  vert->z += v.z ;  vert->nx += v.nx ; vert->ny += v.ny ; vert->nz += v.nz ; }

  vert->x  /= u ;
  vert->y  /= u ;
  vert->z  /= u ;

  u = (float) sqrt( vert->nx * vert->nx + vert->ny * vert->ny +vert->nz * vert->nz ) ;
  if( u > 0 )
  {
    vert->nx /= u ;
    vert->ny /= u ;
    vert->nz /= u ;
  }

  return (mcb->nverts-1) ;
}

static int littleEndianPlatform () {
	uint32_t value = 1;
	return (*((char *) &value) == 1);
}

#ifdef NII2MESH
int marchingCubes(float * img, size_t dim[3], int lo[3], int hi[3], int originalMC, float isolevel, vec3d **vs, vec3i **ts, int *nv, int *nt) {
  MCB * mcp = MarchingCubes(-1, -1, -1);
  int NX = hi[0] - lo[0] + 1;
  int NY = hi[1] - lo[1] + 1;
  int NZ = hi[2] - lo[2] + 1;
  set_resolution( mcp, NX, NY, NZ) ;
  init_all(mcp) ;
  float * im = mcp->data;
  int i = 0;
  int inX = dim[0];
  int inXY = dim[0] * dim[1];
  for (int z=0;z<NZ;z++) //fill voxels
    for (int y=0;y<NY;y++) {
      int zy = ((y+lo[1]) * inX) + ((z+lo[2]) * inXY);
      for (int x=0;x<NX;x++) {
        int j = lo[0] + x + zy;
        im[i] = img[j] - isolevel;
        i++;
      }
    }
  set_method(mcp, originalMC );
  run(mcp) ;
  clean_temps(mcp) ;
  if ((mcp->nverts < 3) || (mcp->ntrigs < 1)) {
    clean_all(mcp);
    free(mcp);
    return EXIT_FAILURE;
  }
  int npt = mcp->nverts;
  *vs = malloc(npt*sizeof(vec3d));
  for (int i = 0; i < npt; i++) {
    (*vs)[i].x = mcp->vertices[i].x + lo[0];
    (*vs)[i].y = mcp->vertices[i].y + lo[1];
    (*vs)[i].z = mcp->vertices[i].z + lo[2];
  }
  int ntri = mcp->ntrigs;
  *ts = malloc(ntri * sizeof(vec3i));
  for (int i=0;i<ntri;i++) {
    (*ts)[i].x = mcp->triangles[i].v3;
    (*ts)[i].y = mcp->triangles[i].v2;
    (*ts)[i].z = mcp->triangles[i].v1;
  }
  *nv = npt; //number of vertices
  *nt = ntri; //number of triangles
  clean_all(mcp);
  free(mcp);
  return EXIT_SUCCESS;
}
#endif //#ifdef NII2MESH

// #ifdef MC_SELF_TEST
//These functions are only used by the self testing executable

void writePLY( MCB *mcb , const char *fn, int startframe, int xval, int yval, int height) {
    float x_offset, y_offset, z_offset, correction;
    correction = (float) height * 125.0/2.0;
  x_offset = (float) xval - correction;
  y_offset = (float) yval - correction;
  z_offset = (float) startframe;
  typedef struct  __attribute__((__packed__)) {
    uint8_t n;
    int32_t x,y,z;
  } vec1b3i;
  typedef struct {
    float x,y,z,nx,ny,nz;
  } vec3s; //single precision (float32)
  int npt = mcb->nverts;
  int ntri = mcb->ntrigs;
  if ((npt < 3) || (ntri < 1)) {
    printf("Unable to create PLY file: No geometry\n");
    return;
  }
  FILE *fp = fopen(fn,"wb");
  if (fp == NULL)
    return;// EXIT_FAILURE;
  fputs("ply\n",fp);
  if (&littleEndianPlatform)
    fputs("format binary_little_endian 1.0\n",fp);
  else
    fputs("format binary_big_endian 1.0\n",fp);
  fputs("comment niimath\n",fp);
  char vpts[80];
  sprintf(vpts, "element vertex %d\n", npt);
  fwrite(vpts, strlen(vpts), 1, fp);
  fputs("property float x\n",fp);
  fputs("property float y\n",fp);
  fputs("property float z\n",fp);
  fputs("property float nx\n",fp);
  fputs("property float ny\n",fp);
  fputs("property float nz\n",fp);
  char vfc[80];
  sprintf(vfc, "element face %d\n", ntri);
  fwrite(vfc, strlen(vfc), 1, fp);
  fputs("property list uchar int vertex_indices\n",fp);
  fputs("end_header\n",fp);
  vec3s *pts32 = (vec3s *) malloc(npt * sizeof(vec3s));
  for (int i = 0; i < npt; i++) { //double->single precision
    pts32[i].x = mcb->vertices[i].x + x_offset;
    pts32[i].y = mcb->vertices[i].y + y_offset;
    pts32[i].z = mcb->vertices[i].z + z_offset;
    pts32[i].nx = mcb->vertices[i].nx;
    pts32[i].ny = mcb->vertices[i].ny;
    pts32[i].nz = mcb->vertices[i].nz;
  }
  fwrite(pts32, npt * sizeof(vec3s), 1, fp);
  free(pts32);
  vec1b3i *tris4 = (vec1b3i *) malloc(ntri * sizeof(vec1b3i));
  for (int i = 0; i < ntri; i++) { //double->single precision
    tris4[i].n = 3;
    tris4[i].x = mcb->triangles[i].v3;
    tris4[i].y = mcb->triangles[i].v2;
    tris4[i].z = mcb->triangles[i].v1;
  }
  fwrite(tris4, ntri * sizeof(vec1b3i), 1, fp);
  free(tris4);
  fclose(fp);
}

typedef struct {
	int sizeof_hdr;
	char ignore[36];
	short dim[8];
	char ignore2[14];
	short datatype, bitpix, slice_start;
	float pixdim[8], vox_offset, scl_slope, scl_inter;
	char ignore3[224];
	char magic[4];
} TNIFTI;



void writeNIFTI(float *img32, int *dim, const char *fn ) {
  TNIFTI hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.sizeof_hdr = 348;
  hdr.dim[0] = 3;
  hdr.dim[1] = dim[0];
  hdr.dim[2] = dim[1];
  hdr.dim[3] = dim[2];
  hdr.datatype = 16; //DT_FLOAT32
  hdr.bitpix = 32;
  hdr.pixdim[1] = 1.0;
  hdr.pixdim[2] = 1.0;
  hdr.pixdim[3] = 1.0;
  hdr.vox_offset = 352;
  hdr.scl_slope = 1.0;
  hdr.scl_inter = 0.0;
  hdr.magic[0] = 'n';
  hdr.magic[1] = '+';
  hdr.magic[2] = '1';
  FILE *fp = fopen( fn, "wb" ) ;
  fwrite(&hdr, sizeof(TNIFTI), 1, fp);
  int32_t dummy = 0;
  fwrite(&dummy, sizeof(dummy), 1, fp);
  int nvox = dim[0] * dim[1] * dim[2];
  fwrite(img32, nvox * sizeof(float), 1, fp);
  fclose(fp);
}

//_____________________________________________________________________________
// Compute data
// compute_data(&data, img32,dim,uniquelist[0]);
void compute_data(int64_t** data, float* img32,int* dim,int64_t uniquelist) {

float val = 0 ;
  int i, j, k,m;

int cols = dim[1];
int slices = dim[2];
  long long vox = 0;

  for(  k = 0 ; k < dim[2] ; k++ )
  {
    
        for(  j = 0 ; j < dim[1] ; j++ )
        {
        //   y = ( (float) j ) / sy  - ty ;
            for(  i = 0 ; i < dim[0] ; i++ )
            {
                if ((*data)[i*slices*cols+j*slices+k] == uniquelist){
                // x = ( (float) i ) / sx - tx ; 
                    val = -1.0;
                }
                else{
                    val = 1.0;
                }
    // if (uniquelist == 864691136698463485){
    //     printf("val %f, i %d, j %d, k %d,slices %d\n",val,i,j,k,slices);
    // }
                img32[vox] = val;
                vox++;
            }
        }

    }
    
  

  return;
}










void analyzeFile(FILE* file, int* rows, int* cols, int* max_chars) {
  *rows = 0;
  *cols = 0;
  *max_chars = 0;
  int line_chars = 0;
  int count = 0;
  char ch;

  while ((ch = fgetc(file)) != EOF) {
    line_chars++;
    if (ch == ',') {
      count++;
    }
    if (ch == '\n') {
      (*rows)++;
      if (line_chars > *max_chars) {
        *max_chars = line_chars;
      }
      if (count > *cols) {
        *cols = count;
      }
      line_chars = 0;
      count = 0;
    }
  }
  (*rows)--;
  // Adjust cols to be number of entries per line
  (*cols)++;
}

void readCSV(const char* currentPath, int* rows, int* cols,int startframe,int xval,int yval,int height) {
    char filename[256];
    char buffer[MAX_LINE_LENGTH];
    int max_chars;
    char* token;
    int entryCount;
    int rows_present;
    int cols_present;
      // Allocate memory for the data array

    
    sprintf(filename, "/../Neuron_transport_data/szmframe_%d_%d_%d_%d.csv", startframe, xval, yval, height);
    
    // sprintf(filename, "/../Neuron_transport_data/frame_%d_%d_%d.csv", startframe, xval, yval);
    strcpy(buffer, currentPath);
    strcat(buffer, filename);

    FILE* file = fopen(buffer, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }


    analyzeFile(file,  rows, cols, &max_chars);
    // printf("Number of rows %d\n", *rows);
    // printf("Number of cols %d\n", *cols);
    // printf("Max characters per line %d\n", max_chars);

    return;
}

void analyzeCSV(const char* currentPath, int64_t** data, int* rows, int* cols,int slices,int startframe,int xval,int yval,int height) {
    char filename[256];
    char buffer[MAX_LINE_LENGTH];
    int max_chars;
    char* token;
    int entryCount;
       int rows_present;
    int cols_present;
    // strcpy(buffer, currentPath);
    // strcat(buffer, filename);
    // printf("reading_CSV\n");
    for (int slice=0;slice < slices;slice++){
        // sprintf(filename, "/../Neuron_transport_data/frame_%d_%d_%d.csv", startframe+slice, xval, yval);
        
        sprintf(filename, "/../Neuron_transport_data/szmframe_%d_%d_%d_%d.csv", startframe+slice, xval, yval,height);
            
        
        strcpy(buffer, currentPath);
        strcat(buffer, filename);

        FILE* file = fopen(buffer, "r");
        if (file == NULL) {
            printf("Failed to open file: %s\n", filename);
            return;
        }

        
        
        analyzeFile(file,  &rows_present, &cols_present, &max_chars);
        // printf("Number of rows %d\n", rows_present);
        // printf("Number of cols %d\n", cols_present);
        // printf("Max characters per line %d\n", max_chars);
        if (rows_present != *rows) {
            printf("Error: Number of rows in file %s does not match the number of rows in the first file\n", filename);
            return;
        }
        if (cols_present != *cols) {
            printf("Error: Number of cols in file %s does not match the number of cols in the first file\n", filename);
            return;
        }


        // Allocate memory for the line buffer
        char* line = (char*)malloc((max_chars + 1) * sizeof(char));

        fseek(file, 0, SEEK_SET);
        // Ignore the first row
        fgets(line, max_chars+1, file);
        // printf("%s",line);

        // Read the data from the CSV file
        for (int i = 0; i < *rows; i++) {
            fgets(line, max_chars+1, file);
            // printf("%s",line);
            if (line[0] == '\n') {
            printf("Empty line\n");
            }
            // printf("%s\n",line);
            char* line_copy = strdup(line);
            token = strtok(line_copy, ",");
            
            (*data)[i*slices*(*cols)+slice]= strtoll(token, NULL, 10);
            // printf("row i=%d %s\n",i,token);
            entryCount = 0;
            for (int j = 1; j < *cols; j++) {
                // printf("col j=%d %s\n",j,token);
                token = strtok(NULL, ",\n");
                if (token == NULL) {
                    printf("Premature end of row %d\n", i+1);
                    break;
                }
                (*data)[i*slices*(*cols)+j*slices+slice] = strtoll(token, NULL, 10);
                
                entryCount++;
                // printf("%d %s\n",entryCount,token);
            }
            // printf("entryCount %d\n",entryCount);
            if (entryCount != *cols-1) {
              printf("Error: Row %d does not contain exactly %d entries\n", i+1, *cols);
            }
            free(line_copy);
        }

        // Free the line buffer
        free(line);
        fclose(file);
    }
    
    
    return;
}


char* get_current_path() {
    static char path[1024];
    if (getcwd(path, sizeof(path)) != NULL) {
        return path;
    } else {
        perror("Error getting current path");
        return NULL;
    }
}





void parse_settings_file(int* Sim_number, int* slices, int* startframe, int* xval, int* yval, int* height, char* currentPath, int settings_number) {
    char buffer[MAX_LINE_LENGTH];
    strcpy(buffer, currentPath);
    strcat(buffer, "/../Neuron_transport_data/MarchingCubes_settings_smooth.csv");
    printf("Opening settings file: %s\n", buffer);
    FILE* file = fopen(buffer, "r");
    if (file == NULL) {
        printf("Error opening settings file.\n");
        return;
    }
    else{
        printf("Settings file opened.\n");
    }

    char line[1024];
    char lastLine[1024];
    char* token;
    // Remove header
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file) != NULL) {
        strcpy(lastLine, line);
        token = strtok(lastLine, ",");
        *Sim_number = atoi(token);
        if (*Sim_number == settings_number) {
            break;
        }
    }

    fclose(file);
    // printf("lastLine: %s\n", lastLine);
    
    // token = strtok(lastLine, ",");
    // *Sim_number = atoi(token);

    token = strtok(NULL, ",");
    *slices = atoi(token)+1;

    token = strtok(NULL, ",");
    *startframe = atoi(token);

    token = strtok(NULL, ",");
    *xval = atoi(token);

    token = strtok(NULL, ",");
    *yval = atoi(token);

    token = strtok(NULL, ",");
    *height = atoi(token);

}

void parse_unique_file(int64_t** uniquelist, int* uniquelistlength, char* currentPath, int slices, int startframe, int xval, int yval, int height) {
  char buffer[MAX_LINE_LENGTH];
  char filename[MAX_LINE_LENGTH];
  strcpy(buffer, currentPath);
  int endframe = startframe + slices -1;
  sprintf(filename, "/../Neuron_transport_data/frame_uniq_%d_%d_%d_%d_%d.csv", startframe, endframe, xval, yval, height);
  strcat(buffer, filename);
  printf("Opening unique file: %s\n", buffer);
  FILE* file = fopen(buffer, "r");
  if (file == NULL) {
      printf("Error opening unique file. %s\n",filename);
      return;
  }
  int ch;
  int max_length = 0;
  int current_length = 0;
  while ((ch = fgetc(file)) != EOF) {
    if (ch == '\n') {
          if (current_length > max_length) {
              max_length = current_length;
          }
          current_length = 0;
      } else {
          current_length++;
      }
  }

  if (current_length > max_length) {
    max_length = current_length;
  }
  // First pass: count the commas
  fseek(file, 0, SEEK_SET);  // Reset file position to the start
  int line_count = 0;
  char line[max_length+1];
  int i = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
    if (strlen(line) > 1) {
      // printf("Line: %s %d\n", line,strlen(line));
      line_count++;
    }
  }
  fseek(file, 0, SEEK_SET);  // Reset file position to the start
  *uniquelistlength = line_count - 1;
  printf("uniquelistlength: %d\n", *uniquelistlength);
  // Allocate memory for uniquelist
  *uniquelist = malloc((*uniquelistlength) * sizeof(long long));
  if (*uniquelist == NULL) {
    printf("Failed to allocate memory for uniquelist\n");
    exit(1);
  }

  // Second pass: read in each item
  fseek(file, 0, SEEK_SET);  // Reset file position to the start
 
  fgets(line, sizeof(line), file); //Remove header
  while (fgets(line, sizeof(line), file) != NULL) {
    // char* token = strtok(line, "\n");
    // printf("Token: %s\n", token);
    // printf("Token: %lld, %d\n", strtoll(token, NULL, 10),sizeof(strtoll(token, NULL, 10)));
    //     /* If the result is 0, test for an error */
    if (strlen(line) > 1) {
        // for (int j = 0; j < strlen(line); j++) {
        //   printf("Character as char: %c\n", line[j]);
        //   printf("Character as int: %d\n", line[j]);
        // }
    


          // printf("Token: %lld, %d\n", strtoll(token, NULL, 10),sizeof(strtoll(token, NULL, 10)));
        (*uniquelist)[i] = strtoll(line, NULL, 10);
        // printf("uniquelist val: %lld\n", (*uniquelist)[i]);
        if ((*uniquelist)[i]  == 0)
        {
          printf("Conversion error occurred: \n");
          exit(0);
        }  
        i++;
    }
  }
  // for (i=0;i<*uniquelistlength;i++) {
  //   printf("uniquelist[%d]: %lld\n",i,(*uniquelist)[i]);
  // }
  
  fclose(file);
}

void run_Marching_Cubes(int* dim, int64_t** data, int64_t uniquelist,int startframe,int xval,int yval,int height){
    // printf("dim[0]: %d dim[1]: %d dim[2]: %d, sizeof(size_t) %d, sizeof(float) %d\n",dim[0],dim[1],dim[2],sizeof(size_t),sizeof(float));
    
    size_t size_img = ((size_t)dim[0]*(size_t)dim[1]*(size_t)dim[2]*(size_t)sizeof(float)); 
    
 
    float* img32 =(float*) malloc(size_img);
    if (img32 == NULL) {
      fprintf(stderr, "Failed to allocate memory for img32.\n");
      exit(1);
    }
    int originalMC = 0;
    compute_data(data, img32,dim,uniquelist);
    // printf("img32[0] = %f\n",img32[0]);
    // void compute_data(int64_t**** data, float* img32, int *dim,long long uniquelist)
    
    MCB *mcp;
    mcp = MarchingCubes(-1, -1, -1);
    set_resolution( mcp, (int)dim[0], (int)dim[1], (int)dim[2]) ;
  
    init_all(mcp);
   
    set_method(mcp, originalMC );
    memcpy(mcp->data, img32,size_img ) ;
    free(img32);
    // double startTime = clockMsec();
    run(mcp);
    printf("output mesh vert: %d tris: %d for %lld\n", mcp->nverts, mcp->ntrigs,uniquelist);
    clean_temps(mcp) ;
    #define MAX_BUF 1024
    char path[MAX_BUF], meshfn[MAX_BUF];
    if (getcwd (path, MAX_BUF) != path) exit(EXIT_FAILURE);
    
    int len = snprintf (meshfn, MAX_BUF-1, "%s/../Neuron_transport_data/%lld_%d_%d_%d_%d.ply", path, uniquelist,startframe,xval,yval,height);
    // printf(  "Assuming write permissions to Save object %lld as PLY %s\n", uniquelist ,meshfn);
  
    if (len < 0) exit(EXIT_FAILURE);
    writePLY(mcp, meshfn, startframe, xval, yval, height);
    clean_all(mcp) ;
    free(mcp);
}

int job_queue(int numJobs,int rank,int size, int* dim, int64_t** data, int64_t** uniquelist,int startframe,int xval,int yval,int height, MPI_Comm comm) {
    // int rank, size, dst;
    
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(comm);
    if (rank == 0) {
        // Root (manager) code
        int dst;
        int numWorkers = size - 1; // Number of workers
        int jobsSent = 0; // Number of jobs sent to workers
        // int responseRcvd = 0; // Number of responses received from workers
        int jobsCompleted = 0; // Number of jobs completed by workers
        int* message_avail = (int*) calloc(numWorkers, sizeof(int)); // Message sent from workers
        int* status_workers = (int*) calloc(numWorkers, sizeof(int)); // Status of workers (0 = idle, 1 = message sent, 2 = rcvd message and busy, 3 = sent result and idle)
        int* flag = (int*) calloc(numWorkers, sizeof(int)); // Flag for MPI_Test
        int* result = (int*) calloc(numWorkers, sizeof(int)); // Result from workers
        MPI_Request* request_workers = (MPI_Request*) calloc(numWorkers, sizeof(MPI_Request)); // Request for workers
        int terminate = -1;
        // Send initial jobs to workers
        for (dst = 1; dst <= numWorkers; dst++) {
            if (jobsSent >= numJobs) break;
            jobsSent++;
            MPI_Isend(&jobsSent, 1, MPI_INT, dst, JOB_TAG+dst, comm, &request_workers[dst-1]);
            // printf("Root: Sent job %d to Worker %d\n", jobsSent, dst);
            status_workers[dst-1] = 1;
        }

        // Receive results from workers and send additional jobs
        while (jobsCompleted < numJobs) {
            for (dst = 1; dst <= numWorkers; dst++) {
                if (status_workers[dst-1] == 1) {
                    MPI_Test(&request_workers[dst-1], &flag[dst-1], MPI_STATUS_IGNORE);
                    // printf("Flag = %d\n",flag[dst-1]);
                    if  (flag[dst-1] == 1) {
                        status_workers[dst-1] = 2;
                        flag[dst-1] = 0;
                        // printf("Root: Worker %d busy\n", dst);
                    }
                }
            }

            // Check for results from workers
            for (dst = 1; dst <= numWorkers; dst++) {
                // printf("Status worker %d = %d\n",dst,status_workers[dst-1]);
                if (status_workers[dst-1] == 2){
                    MPI_Iprobe(dst, RESULT_TAG+dst, comm, &message_avail[dst-1], MPI_STATUS_IGNORE);
                    // printf("Message avail = %d from Worker: %d\n",message_avail[dst-1],dst);
                    if (message_avail[dst-1] != 0) {
                        MPI_Recv(&result[dst-1], 1, MPI_INT, dst, RESULT_TAG+dst, comm, MPI_STATUS_IGNORE);
                        jobsCompleted++;
                        message_avail[dst-1] = 0;
                        // printf("Jobs complete: %d with completion of job: %d from Worker: %d\n", jobsCompleted,result[dst-1],dst);
                       
                        if (jobsSent < numJobs) {
                            jobsSent++;
                            MPI_Isend(&jobsSent, 1, MPI_INT, dst, JOB_TAG+dst, comm, &request_workers[dst-1]);
                            // printf("Root: Sent job %d to Worker %d\n", jobsSent, dst);   
                            status_workers[dst-1] = 1;                 
                        } else {
                            MPI_Isend(&terminate, 1, MPI_INT, dst, JOB_TAG+dst, comm, &request_workers[dst-1]);
                            // printf("Root: Sent termination signal to Worker %d\n", dst);
                        }
                    }
                }
            }
        }
        MPI_Waitall(numWorkers, request_workers, MPI_STATUSES_IGNORE);
        printf("Root: Completed %d jobs \n",jobsCompleted);
    } else {
        // Worker code
        int job = 0,status_worker = 0, message_avail = 0;
        while (job != -1) {
        //   printf("Worker %d: Waiting for job\n", rank);
            if (status_worker == 0) {
                // Check for job from root
                MPI_Iprobe(0, JOB_TAG+rank, comm, &message_avail, MPI_STATUS_IGNORE);
                if (message_avail != 0) {
                    MPI_Recv(&job, 1, MPI_INT, 0, JOB_TAG+rank, comm, MPI_STATUS_IGNORE);
                    // printf("Worker %d: Received job %d\n", rank, job);
                    status_worker = 1;
                    message_avail = 0;
                }
            }
            else  {
        
                
                run_Marching_Cubes(dim, data, (*uniquelist)[job-1],startframe,xval,yval,height);
                status_worker = 0;

                
                // printf("Worker %d: Executed job %d\n", rank, job);
                // Send result back to root
                MPI_Send(&job, 1, MPI_INT, 0, RESULT_TAG+rank, comm);
            
            }
        }
        // printf("Worker %d Finished jobs \n",rank);
    }
    MPI_Barrier(comm);
    // MPI_Finalize();
    return 0;
}

int main(int argc, char** argv)
{

    // // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int numWorkers = size - 1;

    int settings_number = 0;
    if (argc > 1)
    {
        settings_number = atoi(argv[1]);
    }

    if (rank == 0){
        printf("MarchingCubes_settings simulation number: %d\n",settings_number);
    }
    

    int dim[3];
    int64_t *data;
    int rows, cols;
    int Sim_number, slices, startframe, xval, yval, height;
    char *currentPath = get_current_path();
    if (currentPath != NULL  && rank == 0)
    {
        printf("Current path: %s\n", currentPath);
    }
    int64_t *uniquelist;
    int uniquelistlength;
    int frame_parameters[6];

    if (rank == 0)
    {

        // int Sim_number, slices, startframe, xval, yval, height;

        parse_settings_file(&Sim_number, &slices, &startframe, &xval, &yval, &height, currentPath, settings_number);

        parse_unique_file(&uniquelist, &uniquelistlength, currentPath, slices, startframe, xval, yval, height);
        // for (int i = 0; i < uniquelistlength; i++)
        // {
        //     printf("uniquelist[%d]: %lld\n", i, uniquelist[i]);
        // }

        printf("Sim_number: %d\n", Sim_number);
        printf("slices: %d\n", slices);
        printf("startframe: %d\n", startframe);
        printf("xval: %d\n", xval);
        printf("yval: %d\n", yval);
        printf("height: %d\n", height);
        // const char* filename = "/../Neuron_transport_data/frame_17391_170398_61241.csv";

        readCSV(currentPath, &rows, &cols, startframe, xval, yval, height);

        dim[0] = rows;
        dim[1] = cols;
        dim[2] = slices;

     

        frame_parameters[0] = Sim_number;
        frame_parameters[1] = startframe;
        frame_parameters[2] = xval;
        frame_parameters[3] = yval;
        frame_parameters[4] = height;
        frame_parameters[5] = uniquelistlength;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("Broadcast, rank = %d, dim = %d %d %d\n", rank, dim[0], dim[1], dim[2]);
    rows = dim[0];
    cols = dim[1];
    slices = dim[2];
    // printf("rows = %d, cols = %d, slices = %d\n", rows, cols, slices);
    // data = (int64_t***)malloc((rows) * sizeof(int64_t**));
    // printf("data allocated, %p, rank: %d\n",data,rank);
    // for (int i = 0; i < rows; i++) {
    //     (data)[i] = (int64_t**)malloc((cols) * sizeof(int64_t*));
    //     for (int j = 0; j < cols; j++) {
    //     (data)[i][j] = (int64_t*)malloc((slices) * sizeof(int64_t));
    //     }
    // }
    // data = (int64_t*)malloc((rows*cols*slices) * sizeof(int64_t));
    MPI_Alloc_mem((rows * cols * slices) * sizeof(int64_t), MPI_INFO_NULL, &data);
    MPI_Win win;
    // printf("win created\n");
    int null_buffer[1] = {0};
    if (rank == 0)
    {
        MPI_Win_create(data, rows * cols * slices * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        MPI_Win_create(null_buffer, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    } // size_t maxAllocationSize = SIZE_MAX;
    // printf("first win created\n");
    // MPI_Win_create(frame_parameters, 7*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    printf("Broadcasting frame parameters\n");
    MPI_Bcast(&frame_parameters, 6, MPI_INT, 0, MPI_COMM_WORLD);
    Sim_number = frame_parameters[0];
    startframe = frame_parameters[1];
    xval = frame_parameters[2];
    yval = frame_parameters[3];
    height = frame_parameters[4];
    uniquelistlength = frame_parameters[5];
    // printf("Sim_number: %d, rank: %d\n", Sim_number, rank);
    if (rank != 0)
    {
        uniquelist = (int64_t *)malloc((uniquelistlength) * sizeof(int64_t));
    }
    printf("Broadcasting list of unique cells\n");
    MPI_Bcast(uniquelist, uniquelistlength, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // printf("uniquelistlength: %d, rank: %d\n", uniquelistlength, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        analyzeCSV(currentPath, &data, &rows, &cols, slices, startframe, xval, yval, height);
     
    }
    // printf("Finished reading CSV\n");
    MPI_Barrier(MPI_COMM_WORLD);
    //
    MPI_Win_fence(0, win);
    if (rank != 0)
    {

        MPI_Get(data, dim[0] * dim[1] * dim[2], MPI_INT64_T, 0, 0, dim[0] * dim[1] * dim[2], MPI_INT64_T, win);
      
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_fence(0, win);

    // MPI_Barrier(MPI_COMM_WORLD);
    // need: dim,data,unique_list,startframe,xval,yval,height

    // Print the data for verification

    MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank: %d, uniquelistlength: %d\n", rank, uniquelistlength);
    job_queue(uniquelistlength, rank, size, dim, &data, &uniquelist, startframe, xval, yval, height, MPI_COMM_WORLD);

    // Free the allocated memory
    // for (int i = 0; i < rows; i++) {
    //   for (int j = 0; j < cols; j++) {
    //       free(data[i][j]);
    //   }
    // free(data[i]);
    // }
    printf("rank: %d, cleanup\n", rank);
    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank: %d, win free\n", rank);
    MPI_Free_mem(data);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank: %d, data free\n", rank);
    free(uniquelist);

    //  MPI_Barrier(MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank: %d, terminating\n", rank);

    MPI_Finalize();
    // exit(EXIT_SUCCESS);
    return 0;
}
