/*
//@HEADER
// ************************************************************************
//
//                        Adelus v. 1.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of NTESS nor the names of the contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Vinh Dang (vqdang@sandia.gov)
//                    Joseph Kotulski (jdkotul@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef __ADELUS_PERMRHS_HPP__
#define __ADELUS_PERMRHS_HPP__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Adelus_defines.h"
#include "Adelus_macros.h"
#include "Adelus_mytime.hpp"
#include "Kokkos_Core.hpp"

//extern int me;                 // processor id information
//extern int nprocs_row;         // num of procs to which a row is assigned
//extern int nprocs_col;         // num of procs to which a col is assigned
//extern int nrows_matrix;       // number of rows in the matrix
//extern int ncols_matrix;       // number of cols in the matrix
//extern int my_rows;            // num of rows I own
//extern int my_cols;            // num of cols I own
//extern int my_rhs;             // num of right hand side I own
//extern int myrow;
//extern int mycol;
//extern MPI_Comm col_comm;

namespace Adelus {
  
  template<class ZViewType, class PViewType>
  inline
  void permute_rhs(ZViewType& RHS, PViewType& permute, int& my_num_rhs) {
    using value_type      = typename ZViewType::value_type;
    using execution_space = typename ZViewType::device_type::execution_space ;
    using memory_space    = typename ZViewType::device_type::memory_space ;
    using ViewVectorType  = Kokkos::View<value_type*, Kokkos::LayoutLeft, memory_space>;
  
    int pivot_row, k_row;
    ViewVectorType tmpr( "tmpr", RHS.extent(1) );
    ViewVectorType tmps( "tmps", RHS.extent(1) );

    MPI_Status msgstatus;

    //TODO: add host pinned memory support

    //TODO: try this later
    //MPI_Datatype strided_vec_type;
    //int strided_vec_nblocks  = RHS.extent(1);
    //int strided_vec_blocklen = 1;
    //int strided_vec_stride   = RHS.extent(0);
    //MPI_Type_vector( strided_vec_nblocks, strided_vec_blocklen, strided_vec_stride,
    //                 ADELUS_MPI_DATA_TYPE, &strided_vec_type);
    //MPI_Type_commit(&strided_vec_type);

#ifdef GET_TIMING
   double permuterhstime,t1;

   t1 = MPI_Wtime();
#endif

    for (int k=0;k<=nrows_matrix-2;k++) {
      k_row=k%nprocs_col;

      if (my_num_rhs > 0) {
        if (myrow==k_row) pivot_row = permute(k/nprocs_col);
        MPI_Bcast(&pivot_row,1,MPI_INT,k_row,col_comm);
        int pivot_row_pid = pivot_row%nprocs_col;

        if (k != pivot_row) {
          if (k_row == pivot_row_pid) {//pivot row is in the same rank
            if (myrow == k_row) {
              int curr_lrid = k/nprocs_col;
              int piv_lrid  = pivot_row/nprocs_col;
              Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,RHS.extent(1)), KOKKOS_LAMBDA (const int i) {
                value_type tmp   = RHS(curr_lrid,i);
                RHS(curr_lrid,i) = RHS(piv_lrid,i);
                RHS(piv_lrid,i)  = tmp;
              });
              Kokkos::fence();
            }
          }
          else {//pivot row is is a different rank
            if (myrow == k_row) {
              int curr_lrid = k/nprocs_col;
              Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,RHS.extent(1)), KOKKOS_LAMBDA (const int i) {
                tmps(i) = RHS(curr_lrid,i);
              });
              Kokkos::fence();

              MPI_Send(reinterpret_cast<char *>(tmps.data()),RHS.extent(1)*sizeof(value_type),MPI_CHAR,pivot_row_pid,2,col_comm);
              MPI_Recv(reinterpret_cast<char *>(tmpr.data()),RHS.extent(1)*sizeof(value_type),MPI_CHAR,pivot_row_pid,3,col_comm,&msgstatus);

              Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,RHS.extent(1)), KOKKOS_LAMBDA (const int i) {
                RHS(curr_lrid,i) = tmpr(i);
              });
              Kokkos::fence();
            }
            if (myrow == pivot_row_pid) {
              int piv_lrid = pivot_row/nprocs_col;
              Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,RHS.extent(1)), KOKKOS_LAMBDA (const int i) {
                tmps(i) = RHS(piv_lrid,i);
              });
              Kokkos::fence();

              MPI_Recv(reinterpret_cast<char *>(tmpr.data()),RHS.extent(1)*sizeof(value_type),MPI_CHAR,k_row,2,col_comm,&msgstatus);
              MPI_Send(reinterpret_cast<char *>(tmps.data()),RHS.extent(1)*sizeof(value_type),MPI_CHAR,k_row,3,col_comm);

              Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0,RHS.extent(1)), KOKKOS_LAMBDA (const int i) {
                RHS(piv_lrid,i) = tmpr(i);
              });
              Kokkos::fence();
            }
          }//End of pivot row is is a different rank
        }// End of if (k != pivot_row)

      }// End of if (my_num_rhs > 0)

    }// End of for (k=0;k<=nrows_matrix-2;k++)

#ifdef GET_TIMING
    permuterhstime = MPI_Wtime()-t1;

    showtime("Time to permute rhs",&permuterhstime);    
#endif
  }// End of function permute_rhs

}//namespace Adelus

#endif
