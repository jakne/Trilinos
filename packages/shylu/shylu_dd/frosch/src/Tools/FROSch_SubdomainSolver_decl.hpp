// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_SUBDOMAINSOLVER_DECL_HPP
#define _FROSCH_SUBDOMAINSOLVER_DECL_HPP

#include <ShyLU_DDFROSch_config.h>

#include <FROSch_Types.h>
#include <FROSch_Output.h>
#include <FROSch_Timers.h>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_Operator.hpp>
#include <Xpetra_Matrix_decl.hpp>

#include "Amesos2.hpp"

#ifdef HAVE_SHYLU_DDFROSCH_BELOS
#include <BelosXpetraAdapterOperator.hpp>
#include <BelosOperatorT.hpp>
#include <BelosXpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#endif

#ifdef HAVE_SHYLU_DDFROSCH_MUELU
//#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>
#endif

#ifdef HAVE_SHYLU_DDFROSCH_IFPACK2
#include <Ifpack2_Details_OneLevelFactory_decl.hpp>
#endif

#ifdef HAVE_SHYLU_DDFROSCH_THYRA
#include "Stratimikos_LinearSolverBuilder.hpp"
#ifdef HAVE_SHYLU_DDFROSCH_IFPACK2
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#endif
#endif


namespace FROSch {

    template <class SC,
              class LO,
              class GO,
              class NO>
    class OneLevelPreconditioner;

    template<class SC,
             class LO,
             class GO,
             class NO>
    class TwoLevelPreconditioner;

    template<class SC,
             class LO,
             class GO,
             class NO>
    class TwoLevelBlockPreconditioner;

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class SubdomainSolver : public Xpetra::Operator<SC,LO,GO,NO> {

    protected:

        using XMap                        = Xpetra::Map<LO,GO,NO>;
        using XMapPtr                     = RCP<XMap>;
        using ConstXMapPtr                = RCP<const XMap>;
        using XMapPtrVecPtr               = Teuchos::ArrayRCP<XMapPtr>;

        using XMatrix                     = Xpetra::Matrix<SC,LO,GO,NO>;
        using XMatrixPtr                  = RCP<XMatrix>;
        using ConstXMatrixPtr             = RCP<const XMatrix>;

        using TCrsMatrix                  = Tpetra::CrsMatrix<SC,LO,GO,NO>;
        using TCrsMatrixPtr               = RCP<TCrsMatrix>;
        using ConstTCrsMatrixPtr          = RCP<const TCrsMatrix>;

        using TRowMatrix                  = Tpetra::RowMatrix<SC,LO,GO,NO>;
        using TRowMatrixPtr               = RCP<TRowMatrix>;
        using ConstTRowMatrixPtr          = RCP<const TRowMatrix>;

        using XMultiVector                = Xpetra::MultiVector<SC,LO,GO,NO>;
        using ConstXMultiVector           = const Xpetra::MultiVector<SC,LO,GO,NO>;
        using XMultiVectorPtr             = RCP<XMultiVector>;
        using ConstXMultiVectorPtr        = RCP<const XMultiVector>;
        using ConstXMultiVectorPtrVecPtr  = Teuchos::ArrayRCP<ConstXMultiVectorPtr>;

        using TMultiVector                = Tpetra::MultiVector<SC,LO,GO,NO>;
        using TMultiVectorPtr             = RCP<TMultiVector>;

        using XMultiVectorFactory         = MultiVectorFactory<SC,LO,GO,NO>;
        using XMultiVectorFactory         = Xpetra::MultiVectorFactory<SC,LO,GO,NO>;

        using ParameterListPtr            = RCP<Teuchos::ParameterList>;

        using Amesos2SolverTpetraPtr      = RCP<Amesos2::Solver<TCrsMatrix,TMultiVector> >;

#ifdef HAVE_SHYLU_DDFROSCH_MUELU
        using MueLuFactoryPtr             = RCP<MueLu::HierarchyManager<SC,LO,GO,NO> >;
        using MueLuHierarchyPtr           = RCP<MueLu::Hierarchy<SC,LO,GO,NO> >;
#endif

        using UN                          = unsigned;
        using UNVec                       = Teuchos::Array<UN>;
        using UNVecPtr                    = Teuchos::ArrayRCP<UN>;

        using GOVecPtr                    = Teuchos::ArrayRCP<GO>;

    public:

        /*!
        \brief Constructor

        Create the subdomain solver object.

        A subdomain solver might require further Trilinos packages at runtime.
        Availability of required packages is tested and
        an error is thrown if required packages are not part of the build configuration.

        @param k Matrix
        @param parameterList Parameter list
        @param blockCoarseSize
        */
        SubdomainSolver(ConstXMatrixPtr k,
                        ParameterListPtr parameterList,
                        std::string description = "undefined",
                        GOVecPtr blockCoarseSize=Teuchos::null);

        //! Destructor
        virtual ~SubdomainSolver();

        //! Initialize member variables
        virtual int initialize();

        /*!
        \brief Compute/setup this operator/solver

        \pre Routine initialize() has been called and #isInitialized_ is set to \c true.

        @return Integer error code
        */
        virtual int compute();

        /*! \brief Apply subdomain solver to input \c x
         *
         * y = alpha * A^mode * X + beta * Y
         *
         * \param[in] x Input vector
         * \param[out] y result vector
         * \param[in] mode
         */

        /*!
        \brief Computes the operator-multivector application.

        Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
        vary according to the values of \c alpha and \c beta. Specifically
        - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
        - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
        */
        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           Teuchos::ETransp mode = Teuchos::NO_TRANS,
                           SC alpha = Teuchos::ScalarTraits<SC>::one(),
                           SC beta = Teuchos::ScalarTraits<SC>::zero()) const;

        //! Get domain map
        virtual ConstXMapPtr getDomainMap() const;

        //! Get range map
        virtual ConstXMapPtr getRangeMap() const;

        /*!
        \brief Print description of this object to given output stream

        \param out Output stream to be used
        \param Verbosity level used for printing
        */
        virtual void describe(Teuchos::FancyOStream &out,
                              const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const;

        /*!
        \brief Get description of this operator

        \return String describing this operator
        */
        virtual std::string description() const;

        //! @name Access to class members
        //!@{

        //! Get #IsInitialized_
        bool isInitialized() const;

        //! Get #IsComputed_
        bool isComputed() const;

        int resetMatrix(ConstXMatrixPtr k,
                        bool reuseInitialize);

        /*!
        \brief Computes a residual using the operator
        */
        virtual void residual(const XMultiVector & X,
                              const XMultiVector & B,
                              XMultiVector& R) const;

        //!@}

    protected:

        //! Matrix
        ConstXMatrixPtr K_;

        //! Paremter list
        ParameterListPtr ParameterList_;

        //! Description of the solver
        std::string Description_;

        mutable XMultiVectorPtr YTmp_;

        Amesos2SolverTpetraPtr Amesos2SolverTpetra_;

#ifdef HAVE_SHYLU_DDFROSCH_MUELU
        //! Factory to create a MueLu hierarchy
        MueLuFactoryPtr MueLuFactory_;

        //! MueLu hierarchy object
        MueLuHierarchyPtr MueLuHierarchy_;
#endif

#ifdef HAVE_SHYLU_DDFROSCH_BELOS
        RCP<Belos::LinearProblem<SC,Xpetra::MultiVector<SC,LO,GO,NO>,Belos::OperatorT<Xpetra::MultiVector<SC,LO,GO,NO> > > > BelosLinearProblem_;
        RCP<Belos::SolverManager<SC,Xpetra::MultiVector<SC,LO,GO,NO>,Belos::OperatorT<Xpetra::MultiVector<SC,LO,GO,NO> > > > BelosSolverManager_;
#endif

#ifdef HAVE_SHYLU_DDFROSCH_IFPACK2
        RCP<Ifpack2::Preconditioner<SC,LO,GO,NO> > Ifpack2Preconditioner_;
#endif

#ifdef HAVE_SHYLU_DDFROSCH_THYRA
        mutable RCP<Thyra::MultiVectorBase<SC> > ThyraYTmp_;
        RCP<Thyra::LinearOpWithSolveBase<SC> > LOWS_;
#endif

        RCP<TwoLevelBlockPreconditioner<SC,LO,GO,NO> > TLBP;
        RCP<TwoLevelPreconditioner<SC,LO,GO,NO> > TLP;

        bool IsInitialized_ = false;

        //! Flag to indicated whether this subdomain solver has been setup/computed
        bool IsComputed_ = false;
    };

}

#endif
