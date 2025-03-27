
#ifndef _FROSCH_EIGENSOLVER_DECL_HPP
#define _FROSCH_EIGENSOLVER_DECL_HPP

// Tpetra
//#include <Tpetra_KokkosCompat_DefaultNode.hpp>

//Teuchos
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// FROSch
#include "FROSch_EigenSolverOperator_decl.hpp"
#include "FROSch_EigenSolverFactory_fwd_decl.hpp"  // forward declaration


namespace FROSch {


template < typename SC,// = double,
           typename LO,// = int,
           typename GO,// = DefaultGlobalOrdinal,
           typename NO,// = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType, 
           typename OpA,
           typename OpB>
class EigenSolver {

friend class EigenSolverFactory< SC, LO, GO, NO, OpA, OpB >;

public:

  using EigenOperatorPtr = Teuchos::RCP< EigenSolverOperator< SC, LO, GO, NO > >;
  using ParameterListPtr = Teuchos::RCP< Teuchos::ParameterList >;

  // Virtual destructor for memory safety of derived classes.
  virtual ~EigenSolver() {};

  //! Initialize the internal solver
//    virtual int initialize() = 0;

  //! Compute the internal solver
//    virtual int compute() = 0;

  //! Get #IsComputed_
//    bool isComputed() const;

protected:

  // Constructor
  EigenSolver(OpA opA, OpB opB, ParameterListPtr parameterList, Teuchos::RCP< std::vector<SC> > &eigenvalues_ptr, Teuchos::RCP< Teuchos::SerialDenseMatrix<LO,SC> >& eigenvectors_ptr);

  EigenOperatorPtr opA_;
  EigenOperatorPtr opB_;

  ParameterListPtr parameterList_;

//    bool isInitialized_ = false;

//    bool isComputed_ = false;

private:

  // Default constructer is not allowed. We want to enforce that the EigenSolver is set up by a constructor.
  EigenSolver() = delete;


};


} // end namespace FROSch

#endif
