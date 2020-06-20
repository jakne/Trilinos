// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_DefaultComm.hpp"

#include "Thyra_VectorStdOps.hpp"

#include "Tempus_StepperFactory.hpp"
#include "Tempus_StepperSubcyclingModifierBase.hpp"
#include "Tempus_StepperSubcyclingModifierXBase.hpp"
#include "Tempus_StepperSubcyclingObserverBase.hpp"
#include "Tempus_StepperSubcyclingModifierDefault.hpp"
#include "Tempus_StepperSubcyclingModifierXDefault.hpp"
#include "Tempus_StepperSubcyclingObserverDefault.hpp"

#include "../TestModels/SinCosModel.hpp"
#include "../TestUtils/Tempus_ConvergenceTestUtils.hpp"

#include <fstream>
#include <vector>

namespace Tempus_Unit_Test {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;
using Teuchos::ParameterList;
using Teuchos::sublist;
using Teuchos::getParametersFromXmlFile;

using Tempus::StepperFactory;
using Tempus::StepperExplicitRK;


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(Subcycling, Default_Construction)
{
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());

  // Default construction.
  auto stepper = rcp(new Tempus::StepperSubcycling<double>());
  auto sf = Teuchos::rcp(new Tempus::StepperFactory<double>());
  auto stepperBE = sf->createStepperBackwardEuler(model, Teuchos::null);
  stepper->setSubcyclingStepper(stepperBE);
  stepper->initialize();
  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());

  // Default values for construction.
#ifndef TEMPUS_HIDE_DEPRECATED_CODE
  auto obs       = rcp(new Tempus::StepperSubcyclingObserver<double>());
#endif
  auto modifier  = rcp(new Tempus::StepperSubcyclingModifierDefault<double>());
  auto modifierX = rcp(new Tempus::StepperSubcyclingModifierXDefault<double>());
  auto observer  = rcp(new Tempus::StepperSubcyclingObserverDefault<double>());
  auto solver    = rcp(new Thyra::NOXNonlinearSolver());
  solver->setParameterList(Tempus::defaultSolverParameters());

  bool useFSAL              = stepper->getUseFSALDefault();
  std::string ICConsistency = stepper->getICConsistencyDefault();
  bool ICConsistencyCheck   = stepper->getICConsistencyCheckDefault();

  // Test the set functions
  stepper->setSolver(solver);                          stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
#ifndef TEMPUS_HIDE_DEPRECATED_CODE
  stepper->setObserver(obs);                           stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
#endif
  stepper->setAppAction(modifier);                     stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
  stepper->setAppAction(modifierX);                    stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
  stepper->setAppAction(observer);                     stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
  stepper->setUseFSAL(useFSAL);                        stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
  stepper->setICConsistency(ICConsistency);            stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
  stepper->setICConsistencyCheck(ICConsistencyCheck);  stepper->initialize();  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());


  // Full argument list construction.
  auto scIntegrator = Teuchos::rcp(new Tempus::IntegratorBasic<double>());
  auto stepperFE = sf->createStepperForwardEuler(model, Teuchos::null);
  scIntegrator->setStepperWStepper(stepperFE);
  scIntegrator->initialize();

#ifndef TEMPUS_HIDE_DEPRECATED_CODE
  stepper = rcp(new Tempus::StepperSubcycling<double>(
    model, obs, scIntegrator, useFSAL, ICConsistency, ICConsistencyCheck));
  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());
#endif

  stepper = rcp(new Tempus::StepperSubcycling<double>(
    model,scIntegrator, useFSAL, ICConsistency, ICConsistencyCheck,modifier));
  TEUCHOS_TEST_FOR_EXCEPT(!stepper->isInitialized());

  // Test stepper properties.
  TEUCHOS_ASSERT(stepper->getOrder() == 1);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(Subcycling, MaxTimeStepDoesNotChangeDuring_takeStep)
{
  // Setup the stepper ----------------------------------------
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());
  auto stepper = rcp(new Tempus::StepperSubcycling<double>());
  auto sf = Teuchos::rcp(new Tempus::StepperFactory<double>());
  auto stepperBE = sf->createStepperBackwardEuler(model, Teuchos::null);
  stepper->setSubcyclingStepper(stepperBE);
  stepper->initialize();

  // Setup SolutionHistory ------------------------------------
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =
    stepper->getModel()->getNominalValues();
  auto icSolution =rcp_const_cast<Thyra::VectorBase<double> >(inArgsIC.get_x());
  auto icState = Tempus::createSolutionStateX(icSolution);
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());
  solutionHistory->addState(icState);
  solutionHistory->initWorkingState();

  // Test
  stepper->setSubcyclingMaxTimeStep(0.5);
  double maxTimeStep_Set = stepper->getSubcyclingMaxTimeStep();
  stepper->takeStep(solutionHistory);
  double maxTimeStep_After = stepper->getSubcyclingMaxTimeStep();

  TEST_FLOATING_EQUALITY(maxTimeStep_Set, maxTimeStep_After, 1.0e-14 );
}


// ************************************************************
// ************************************************************
class StepperSubcyclingModifierTest
  : virtual public Tempus::StepperSubcyclingModifierBase<double>
{
public:

  /// Constructor
  StepperSubcyclingModifierTest()
    : testBEGIN_STEP(false), testEND_STEP(false),
      testCurrentValue(-0.99), testWorkingValue(-0.99),
      testDt(1.5), testType("")
  {}
  /// Destructor
  virtual ~StepperSubcyclingModifierTest(){}

  /// Modify Subcycling Stepper at action location.
  virtual void modify(
    Teuchos::RCP<Tempus::SolutionHistory<double> > sh,
    Teuchos::RCP<Tempus::StepperSubcycling<double> > stepper,
    const typename Tempus::StepperSubcyclingAppAction<double>::ACTION_LOCATION actLoc)
  {
    switch(actLoc) {
    case StepperSubcyclingAppAction<double>::BEGIN_STEP:
    {
      testBEGIN_STEP = true;
      auto x = sh->getCurrentState()->getX();
      testCurrentValue = get_ele(*(x), 0);
      testType = "Subcycling - Modifier";
      stepper->setStepperType(testType);
      break;
    }
    case StepperSubcyclingAppAction<double>::END_STEP:
    {
      testEND_STEP = true;
      auto x = sh->getWorkingState()->getX();
      testWorkingValue = get_ele(*(x), 0);
      testDt = sh->getWorkingState()->getTimeStep()/10.0;
      sh->getWorkingState()->setTimeStep(testDt);
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - unknown action location.\n");
    }
  }

  bool testBEGIN_STEP;
  bool testEND_STEP;
  double testCurrentValue;
  double testWorkingValue;
  double testDt;
  std::string testType;
};

TEUCHOS_UNIT_TEST(Subcycling, AppAction_Modifier)
{
  // Setup the SinCosModel ------------------------------------
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());

  // Setup Stepper for field solve ----------------------------
  auto stepper = rcp(new Tempus::StepperSubcycling<double>());
  auto sf = Teuchos::rcp(new Tempus::StepperFactory<double>());
  auto stepperFE = sf->createStepperForwardEuler(model, Teuchos::null);
  auto modifier = rcp(new StepperSubcyclingModifierTest());
  stepper->setAppAction(modifier);
  stepper->setSubcyclingStepper(stepperFE);

  stepper->setSubcyclingMinTimeStep      (15);
  stepper->setSubcyclingInitTimeStep     (15.0);
  stepper->setSubcyclingMaxTimeStep      (15.0);
  stepper->setSubcyclingStepType         ("Constant");
  stepper->setSubcyclingMaxFailures      (10);
  stepper->setSubcyclingMaxConsecFailures(5);
  stepper->setSubcyclingScreenOutputIndexInterval(1);
  stepper->setSubcyclingPrintDtChanges(true);
  stepper->initialize();

  // Setup TimeStepControl ------------------------------------
  auto timeStepControl = rcp(new Tempus::TimeStepControl<double>());
  timeStepControl->setStepType ("Constant");
  timeStepControl->setInitIndex(0);
  timeStepControl->setInitTime (0.0);
  timeStepControl->setFinalTime(1.0);
  timeStepControl->setInitTimeStep(15.0);
  timeStepControl->initialize();

  // Setup initial condition SolutionState --------------------
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =
    stepper->getModel()->getNominalValues();
  auto icSolution = rcp_const_cast<Thyra::VectorBase<double> > (inArgsIC.get_x());
  auto icState = Tempus::createSolutionStateX(icSolution);
  icState->setTime    (timeStepControl->getInitTime());;
  icState->setIndex   (timeStepControl->getInitIndex());
  icState->setTimeStep(0.0);  // dt for ICs are indicated by zero.
  icState->setSolutionStatus(Tempus::Status::PASSED);  // ICs are passing.

  // Setup SolutionHistory ------------------------------------
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());
  solutionHistory->setName("Forward States");
  solutionHistory->setStorageType(Tempus::STORAGE_TYPE_STATIC);
  solutionHistory->setStorageLimit(2);
  solutionHistory->addState(icState);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  solutionHistory->getWorkingState()->setTimeStep(15.0);
  stepper->takeStep(solutionHistory);

  // Testing that each ACTION_LOCATION has been called.
  TEST_COMPARE(modifier->testBEGIN_STEP, ==, true);
  TEST_COMPARE(modifier->testEND_STEP, ==, true);

  // Testing that values can be set through the Modifier.
  auto x = solutionHistory->getCurrentState()->getX();
  TEST_FLOATING_EQUALITY(modifier->testCurrentValue, get_ele(*(x), 0), 1.0e-15);
  x = solutionHistory->getWorkingState()->getX();
  TEST_FLOATING_EQUALITY(modifier->testWorkingValue, get_ele(*(x), 0), 1.0e-15);
  auto Dt = solutionHistory->getWorkingState()->getTimeStep();
  TEST_FLOATING_EQUALITY(modifier->testDt, Dt, 1.0e-15);

  TEST_COMPARE(modifier->testType, ==, "Subcycling - Modifier");
}

// ************************************************************
// ************************************************************
class StepperSubcyclingObserverTest
  : virtual public Tempus::StepperSubcyclingObserverBase<double>
{
public:

  /// Constructor
  StepperSubcyclingObserverTest()
    : testBEGIN_STEP(false), testEND_STEP(false),
      testCurrentValue(-0.99), testWorkingValue(-0.99),
      testDt(15.0), testType("Subcyling")
  {}

  /// Destructor
  virtual ~StepperSubcyclingObserverTest(){}

  /// Observe Subcycling Stepper at action location.
  virtual void observe(
    Teuchos::RCP<const Tempus::SolutionHistory<double> > sh,
    Teuchos::RCP<const Tempus::StepperSubcycling<double> > stepper,
    const typename Tempus::StepperSubcyclingAppAction<double>::ACTION_LOCATION actLoc)
  {
    switch(actLoc) {
    case StepperSubcyclingAppAction<double>::BEGIN_STEP:
    {
      testBEGIN_STEP = true;
      auto x = sh->getCurrentState()->getX();
      testCurrentValue = get_ele(*(x), 0);
      break;
    }
    case StepperSubcyclingAppAction<double>::END_STEP:
    {
      testEND_STEP = true;
      auto x = sh->getWorkingState()->getX();
      testWorkingValue = get_ele(*(x), 0);
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - unknown action location.\n");
    }
  }

  bool testBEGIN_STEP;
  bool testEND_STEP;
  double testCurrentValue;
  double testWorkingValue;
  double testDt;
  std::string testType;
};

TEUCHOS_UNIT_TEST(Subcycling, AppAction_Observer)
{
  // Setup the SinCosModel ------------------------------------
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());

  // Setup Stepper for field solve ----------------------------
  auto stepper = rcp(new Tempus::StepperSubcycling<double>());
  auto sf = Teuchos::rcp(new Tempus::StepperFactory<double>());
  auto stepperFE = sf->createStepperForwardEuler(model, Teuchos::null);
  auto observer = rcp(new StepperSubcyclingObserverTest());
  stepper->setAppAction(observer);
  stepper->setSubcyclingStepper(stepperFE);

  stepper->setSubcyclingMinTimeStep      (15);
  stepper->setSubcyclingInitTimeStep     (15.0);
  stepper->setSubcyclingMaxTimeStep      (15.0);
  stepper->setSubcyclingStepType         ("Constant");
  stepper->setSubcyclingMaxFailures      (10);
  stepper->setSubcyclingMaxConsecFailures(5);
  stepper->setSubcyclingScreenOutputIndexInterval(1);
  stepper->setSubcyclingPrintDtChanges(true);
  stepper->initialize();

  // Setup TimeStepControl ------------------------------------
  auto timeStepControl = rcp(new Tempus::TimeStepControl<double>());
  timeStepControl->setStepType ("Constant");
  timeStepControl->setInitIndex(0);
  timeStepControl->setInitTime (0.0);
  timeStepControl->setFinalTime(1.0);
  timeStepControl->setInitTimeStep(15.0);
  timeStepControl->initialize();

  // Setup initial condition SolutionState --------------------
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =
    stepper->getModel()->getNominalValues();
  auto icSolution = rcp_const_cast<Thyra::VectorBase<double> > (inArgsIC.get_x());
  auto icState = Tempus::createSolutionStateX(icSolution);
  icState->setTime    (timeStepControl->getInitTime());;
  icState->setIndex   (timeStepControl->getInitIndex());
  icState->setTimeStep(0.0);  // dt for ICs are indicated by zero.
  icState->setSolutionStatus(Tempus::Status::PASSED);  // ICs are passing.

  // Setup SolutionHistory ------------------------------------
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());
  solutionHistory->setName("Forward States");
  solutionHistory->setStorageType(Tempus::STORAGE_TYPE_STATIC);
  solutionHistory->setStorageLimit(2);
  solutionHistory->addState(icState);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  solutionHistory->getWorkingState()->setTimeStep(15.0);
  stepper->takeStep(solutionHistory);

  // Testing that each ACTION_LOCATION has been called.
  TEST_COMPARE(observer->testBEGIN_STEP, ==, true);
  TEST_COMPARE(observer->testEND_STEP, ==, true);

  // Testing that values can be observed through the observer.
  auto x = solutionHistory->getCurrentState()->getX();
  TEST_FLOATING_EQUALITY(observer->testCurrentValue, get_ele(*(x), 0), 1.0e-15);
  x = solutionHistory->getWorkingState()->getX();
  TEST_FLOATING_EQUALITY(observer->testWorkingValue, get_ele(*(x), 0), 1.0e-15);
  TEST_FLOATING_EQUALITY(observer->testDt, 15.0, 1.0e-15);

  TEST_COMPARE(observer->testType, ==, "Subcyling");
}

  // ************************************************************
  // ************************************************************
class StepperSubcyclingModifierXTest
  : virtual public Tempus::StepperSubcyclingModifierXBase<double>
{
public:

  /// Constructor
  StepperSubcyclingModifierXTest()
    : testX_BEGIN_STEP(false), testXDOT_END_STEP(false),
      testX(-0.99), testXDot(-0.99),
      testDt(1.5), testTime(1.5)
  {}

  /// Destructor
  virtual ~StepperSubcyclingModifierXTest(){}

  /// Modify Subcycling Stepper at action location.
  virtual void modify(
    Teuchos::RCP<Thyra::VectorBase<double> > x,
    const double time, const double dt,
    const typename Tempus::StepperSubcyclingModifierXBase<double>::MODIFIER_TYPE modType)
  {
    switch(modType) {
    case StepperSubcyclingModifierXBase<double>::X_BEGIN_STEP:
    {
      testX_BEGIN_STEP = true;
      testX = get_ele(*(x), 0);
      testDt = dt;
      break;
    }
    case StepperSubcyclingModifierXBase<double>::XDOT_END_STEP:
    {
      testXDOT_END_STEP = true;
      testXDot = get_ele(*(x), 0);
      testTime = time;
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - unknown action location.\n");
    }
  }

  bool testX_BEGIN_STEP;
  bool testXDOT_END_STEP;
  double testX;
  double testXDot;
  double testDt;
  double testTime;
};

TEUCHOS_UNIT_TEST(Subcycling, AppAction_ModifierX)
{
  // Setup the SinCosModel ------------------------------------
  auto model   = rcp(new Tempus_Test::SinCosModel<double>());

  // Setup Stepper for field solve ----------------------------
  auto stepper = rcp(new Tempus::StepperSubcycling<double>());
  auto sf = Teuchos::rcp(new Tempus::StepperFactory<double>());
  auto stepperFE = sf->createStepperForwardEuler(model, Teuchos::null);
  auto modifierX = rcp(new StepperSubcyclingModifierXTest());
  stepper->setAppAction(modifierX);
  stepper->setSubcyclingStepper(stepperFE);

  stepper->setSubcyclingMinTimeStep      (15);
  stepper->setSubcyclingInitTimeStep     (15.0);
  stepper->setSubcyclingMaxTimeStep      (15.0);
  stepper->setSubcyclingStepType         ("Constant");
  stepper->setSubcyclingMaxFailures      (10);
  stepper->setSubcyclingMaxConsecFailures(5);
  stepper->setSubcyclingScreenOutputIndexInterval(1);
  stepper->setSubcyclingPrintDtChanges(true);
  stepper->initialize();

  // Setup TimeStepControl ------------------------------------
  auto timeStepControl = rcp(new Tempus::TimeStepControl<double>());
  timeStepControl->setStepType ("Constant");
  timeStepControl->setInitIndex(0);
  timeStepControl->setInitTime (0.0);
  timeStepControl->setFinalTime(1.0);
  timeStepControl->setInitTimeStep(15.0);
  timeStepControl->initialize();

  // Setup initial condition SolutionState --------------------
  Thyra::ModelEvaluatorBase::InArgs<double> inArgsIC =
    stepper->getModel()->getNominalValues();
  auto icSolution = rcp_const_cast<Thyra::VectorBase<double> > (inArgsIC.get_x());
  auto icSolutionDot = rcp_const_cast<Thyra::VectorBase<double> > (inArgsIC.get_x_dot());
  auto icState = Tempus::createSolutionStateX(icSolution,icSolutionDot);
  icState->setTime    (timeStepControl->getInitTime());;
  icState->setIndex   (timeStepControl->getInitIndex());
  icState->setTimeStep(0.0);  // dt for ICs are indicated by zero.
  icState->setSolutionStatus(Tempus::Status::PASSED);  // ICs are passing.

  // Setup SolutionHistory ------------------------------------
  auto solutionHistory = rcp(new Tempus::SolutionHistory<double>());
  solutionHistory->setName("Forward States");
  solutionHistory->setStorageType(Tempus::STORAGE_TYPE_STATIC);
  solutionHistory->setStorageLimit(2);
  solutionHistory->addState(icState);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  solutionHistory->getWorkingState()->setTimeStep(15.0);
  stepper->takeStep(solutionHistory);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  solutionHistory->getWorkingState()->setTimeStep(15.0);
  stepper->takeStep(solutionHistory);

  // Testing that each ACTION_LOCATION has been called.
  TEST_COMPARE(modifierX->testX_BEGIN_STEP, ==, true);
  TEST_COMPARE(modifierX->testXDOT_END_STEP, ==, true);

  // Testing that values can be set through the Modifier.
  auto x = solutionHistory->getCurrentState()->getX();
  TEST_FLOATING_EQUALITY(modifierX->testX, get_ele(*(x), 0), 1.0e-15);
  // Temporary memory for xDot is not guarranteed to exist outside the Stepper.
  auto xDot = stepper->getStepperXDot(solutionHistory->getWorkingState());
  TEST_FLOATING_EQUALITY(modifierX->testXDot, get_ele(*(xDot), 0),1.0e-15);
  auto Dt = solutionHistory->getWorkingState()->getTimeStep();
  TEST_FLOATING_EQUALITY(modifierX->testDt, Dt, 1.0e-15);

  auto time = solutionHistory->getWorkingState()->getTime();
  TEST_FLOATING_EQUALITY(modifierX->testTime, time, 1.0e-15);
}

} // namespace Tempus_Test
