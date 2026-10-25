/**
 * agentic_node.mjs — Chapter 34: Agentic AI Orchestration in Node.js (ES2022+)
 * Demonstrates:
 *   1. Progressive Disclosure Skill Registry (SKILL.md catalog vs on-demand load)
 *   2. Isolated Context Subagents running in parallel via Promise.all()
 *   3. Deterministic Verifier / Reflection Gate before Human Approval
 *
 * Run: node /workspace/book/code/ch34/agentic_node.mjs
 */

const SKILLS_CATALOG = [
  {
    name: "rfq-quote-drafter",
    whenToUse: "Use when preparing multi-line electronic component RFQ quotes, checking BOM stock, or enforcing margin floors.",
    body: `# Skill: RFQ Quote Drafter (SOP-402)
1. Delegate BOM stock & cross-reference lookup to tech_bom_specialist.
2. Delegate P50/P90 pricing to pricing_specialist.
3. Enforce strict floor: unit_price >= P90 * 1.04.
4. Return structured JSON for Sarah's approval gate.`
  },
  {
    name: "quality-claim-rma",
    whenToUse: "Use when a customer reports defective batches or requests an RMA.",
    body: `# Skill: Quality Claim & RMA (SOP-510)\nCheck lot traceability and quarantine stock.`
  }
];

function selectAndLoadSkill(query) {
  const catalogTokens = SKILLS_CATALOG.reduce(
    (acc, s) => acc + Math.ceil((s.name.length + s.whenToUse.length) / 4),
    0
  );
  const match = SKILLS_CATALOG.find((s) =>
    query.toLowerCase().includes("rfq") || query.toLowerCase().includes("quote")
  );
  const loadedTokens = match ? Math.ceil(match.body.length / 4) : 0;
  return {
    skillName: match?.name ?? null,
    catalogTokens,
    loadedTokens,
    totalPromptTokens: catalogTokens + loadedTokens
  };
}

// Simulated isolated subagent execution (clean context window per specialist)
async function runSubagent(role, allowedTools, brief) {
  const start = performance.now();
  // Each subagent only sees its own brief + raw tool outputs (never polluting the parent)
  if (role === "tech_bom_specialist") {
    if (!allowedTools.includes("query_inventory")) {
      throw new Error(`Policy violation: ${role} cannot call query_inventory`);
    }
    const rawLogsTokens = 820; // stays inside subagent context!
    const summary = brief.mpns.map((mpn) => ({
      mpn,
      stock: mpn === "STM32F407VGT6" ? 540 : 12000,
      alternate: mpn === "STM32F407VGT6" ? "STM32F407VET6" : null
    }));
    return {
      role,
      rawSubagentTokens: rawLogsTokens,
      distilledTokens: 55,
      elapsedMs: +(performance.now() - start).toFixed(2),
      output: summary
    };
  }

  if (role === "pricing_specialist") {
    const rawHistoryTokens = 640;
    const quotes = brief.lines.map(({ mpn, qty }) => {
      const p50 = mpn === "STM32F407VGT6" ? 6.45 : 1.02;
      const p90 = mpn === "STM32F407VGT6" ? 7.10 : 1.05;
      const floor = +(p90 * 1.04).toFixed(3);
      // Intentionally propose a slightly low price on LM2596S-ADJ to trigger Reflection
      const proposed = mpn === "LM2596S-ADJ" ? 1.059 : +(p50 * 1.18).toFixed(3);
      return { mpn, qty, p50, p90, floor, unitPrice: proposed };
    });
    return {
      role,
      rawSubagentTokens: rawHistoryTokens,
      distilledTokens: 62,
      elapsedMs: +(performance.now() - start).toFixed(2),
      output: quotes
    };
  }
}

function reflectionVerifier(pricingLines) {
  const violations = [];
  const corrected = pricingLines.map((line) => {
    if (line.unitPrice < line.floor) {
      violations.push(
        `BELOW_FLOOR on ${line.mpn}: proposed ${line.unitPrice} < floor ${line.floor}`
      );
      return { ...line, unitPrice: +(line.floor * 1.05).toFixed(3), selfCorrected: true };
    }
    return { ...line, selfCorrected: false };
  });
  return { passedFirstTry: violations.length === 0, violations, corrected };
}

async function main() {
  const skillInfo = selectAndLoadSkill("Prepare RFQ quote for NordSensor BOM");

  // Run specialist subagents in parallel with isolated contexts
  const [bomRes, priceRes] = await Promise.all([
    runSubagent("tech_bom_specialist", ["query_inventory", "find_alternates"], {
      mpns: ["STM32F407VGT6", "LM2596S-ADJ"]
    }),
    runSubagent("pricing_specialist", ["predict_price"], {
      lines: [
        { mpn: "STM32F407VGT6", qty: 1000 },
        { mpn: "LM2596S-ADJ", qty: 2000 }
      ]
    })
  ]);

  const reflection = reflectionVerifier(priceRes.output);
  const report = {
    runtime: `Node.js ${process.version}`,
    skillProgressiveDisclosure: skillInfo,
    parallelSubagents: [
      { role: bomRes.role, rawTokensIsolated: bomRes.rawSubagentTokens, distilledTokens: bomRes.distilledTokens },
      { role: priceRes.role, rawTokensIsolated: priceRes.rawSubagentTokens, distilledTokens: priceRes.distilledTokens }
    ],
    reflectionGate: reflection,
    status: "READY_FOR_HUMAN_APPROVAL"
  };

  console.log(JSON.stringify(report, null, 2));
}

main();
