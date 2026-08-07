import { summarizeBackendLedger } from "./claw-chat-current-fixture-backend-ledger.mjs";
import { buildAssertionContext } from "./claw-chat-current-fixture-assertion-context.mjs";
import { buildCommonAssertions } from "./claw-chat-current-fixture-common-assertions.mjs";
import { buildNotApplicableAssertions } from "./claw-chat-current-fixture-not-applicable-assertions.mjs";
import { buildScenarioAssertions } from "./claw-chat-current-fixture-scenario-assertions.mjs";
import { assert } from "./claw-chat-current-fixture-utils.mjs";
import { buildGateBContractAssertions } from "./claw-chat-current-fixture-gate-b-contract.mjs";
import {
  buildTurnPlanUpdateScenarioAssertions,
  TURN_PLAN_UPDATE_SCENARIO,
} from "./claw-chat-current-fixture-turn-plan-update.mjs";

export function buildFixtureAssertionReport(input) {
  const backendSummary = summarizeBackendLedger(input.backendLedger);
  const context = buildAssertionContext(input);
  const isHookFixture = input.options?.hookFixture === true;
  const isTurnPlanUpdateScenario =
    input.options?.scenario === TURN_PLAN_UPDATE_SCENARIO && !isHookFixture;
  const isFocusedRuntimeScenario = isTurnPlanUpdateScenario || isHookFixture;
  const commonAssertions = isFocusedRuntimeScenario
    ? {}
    : buildCommonAssertions(context);
  const scenarioAssertions = isHookFixture
    ? {}
    : isTurnPlanUpdateScenario
      ? buildTurnPlanUpdateScenarioAssertions(context)
      : buildScenarioAssertions(context);
  const notApplicableAssertions = isFocusedRuntimeScenario
    ? []
    : buildNotApplicableAssertions(context);
  const gateBContractAssertions = buildGateBContractAssertions(
    context.gateBContract,
  );
  const assertions = {
    ...gateBContractAssertions,
    ...commonAssertions,
    ...scenarioAssertions,
  };

  for (const [key, passed] of Object.entries(assertions)) {
    const evidence =
      key === "identityConsistent"
        ? context.gateBContract.identity
        : key === "explicitTerminalOrPending"
          ? context.gateBContract.outcome
          : null;
    assert(
      passed,
      `断言失败: ${key}${evidence ? `; evidence=${JSON.stringify(evidence)}` : ""}`,
    );
  }

  return {
    appServerRequestMethods: context.appServerRequestMethods,
    backendSummary,
    gateBContract: context.gateBContract,
    gateBContractAssertions,
    assertions,
    commonAssertions,
    scenarioAssertions,
    notApplicableAssertions,
  };
}
