<?php

declare(strict_types=1);

namespace App\Services\Agentic;

use Illuminate\Http\Client\Pool;
use Illuminate\Support\Facades\Http;

/**
 * Chapter 34 — Agentic AI Orchestrator in PHP 8.3 / Laravel
 *
 * Demonstrates how an enterprise Laravel ERP backend acts as the
 * "Brain + Hands + Team of Specialists" orchestrator on top of a
 * private vLLM server (OpenAI-compatible API):
 *   1. Progressive Disclosure: Loads SKILL.md only when needed.
 *   2. Parallel Subagents: Uses Http::pool() for concurrent isolated context windows.
 *   3. Reflection & Policy Gate: Deterministic margin floor check before Sarah's approval.
 */
final class AgenticQuoteOrchestrator
{
    public function __construct(
        private readonly string $vllmBaseUrl = 'http://llm.internal.example:8000/v1',
        private readonly string $model = 'Qwen/Qwen2.5-14B-Instruct',
    ) {}

    /**
     * Load only the relevant SKILL.md body (Progressive Disclosure).
     */
    public function loadSkillBody(string $skillName): string
    {
        $path = base_path("skills/{$skillName}/SKILL.md");
        return is_file($path) ? file_get_contents($path) : '';
    }

    /**
     * Dispatch two specialist subagents in parallel with isolated context windows,
     * then verify the output through a deterministic Reflection gate.
     */
    public function prepareQuoteDraft(string $customer, array $bomLines): array
    {
        $skillSop = $this->loadSkillBody('rfq-quote-drafter');

        // 1. Concurrent Subagent calls (each gets a clean, isolated messages[] array!)
        $responses = Http::pool(fn (Pool $pool) => [
            'bom_specialist' => $pool->as('bom_specialist')
                ->timeout(15)
                ->post("{$this->vllmBaseUrl}/chat/completions", [
                    'model' => $this->model,
                    'temperature' => 0.0,
                    'messages' => [
                        ['role' => 'system', 'content' => "You are tech_bom_specialist. Return only JSON stock & alternate summary."],
                        ['role' => 'user', 'content' => json_encode(['lines' => $bomLines], JSON_UNESCAPED_UNICODE)],
                    ],
                ]),

            'pricing_specialist' => $pool->as('pricing_specialist')
                ->timeout(15)
                ->post("{$this->vllmBaseUrl}/chat/completions", [
                    'model' => $this->model,
                    'temperature' => 0.0,
                    'messages' => [
                        ['role' => 'system', 'content' => "You are pricing_specialist.\n{$skillSop}"],
                        ['role' => 'user', 'content' => json_encode(['customer' => $customer, 'lines' => $bomLines], JSON_UNESCAPED_UNICODE)],
                    ],
                ]),
        ]);

        $bomSummary = $responses['bom_specialist']->json('choices.0.message.content', '[]');
        $pricingRaw = json_decode(
            $responses['pricing_specialist']->json('choices.0.message.content', '[]'),
            true
        ) ?: [];

        // 2. Reflection & Deterministic Financial Policy Gate (unit_price >= P90 * 1.04)
        $violations = [];
        foreach ($pricingRaw as &$line) {
            $floor = round(((float) ($line['p90'] ?? 0.0)) * 1.04, 3);
            if (((float) ($line['unit_price'] ?? 0.0)) < $floor) {
                $violations[] = "BELOW_FLOOR:{$line['mpn']} ({$line['unit_price']} < {$floor})";
                $line['unit_price'] = round($floor * 1.05, 3);
                $line['self_corrected'] = true;
            }
        }
        unset($line);

        return [
            'customer' => $customer,
            'bom_summary' => json_decode($bomSummary, true),
            'pricing_lines' => $pricingRaw,
            'reflection_violations' => $violations,
            'status' => 'PENDING_HUMAN_APPROVAL',
        ];
    }
}
