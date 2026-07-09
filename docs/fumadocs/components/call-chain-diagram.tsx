type CallChainStep = {
  title: string;
  detail?: string;
  role?: string;
};

const defaultSteps: CallChainStep[] = [
  {
    title: 'Catalogs',
    detail: 'Benchmark Zoo, Task YAML, Model Zoo',
    role: 'Source',
  },
  {
    title: 'CLI',
    detail: 'worldfoundry-eval run / evaluate',
    role: 'Entry',
  },
  {
    title: 'Evaluation runner',
    detail: 'Requests, suites, resume, cache',
  },
  {
    title: 'Model runner',
    detail: 'Request normalization and backend selection',
    role: 'Model',
  },
  {
    title: 'Pipeline + operator',
    detail: 'Input shaping and inference call',
    role: 'Runtime',
  },
  {
    title: 'Artifacts',
    detail: 'Video, geometry, traces, structured rows',
  },
  {
    title: 'Benchmark runner',
    detail: 'Metrics, normalizers, official facades',
    role: 'Benchmark',
  },
  {
    title: 'Scorecard',
    detail: 'Reports, blockers, leaderboard eligibility',
    role: 'Report',
  },
];

export function CallChainDiagram({
  steps = defaultSteps,
}: {
  steps?: CallChainStep[];
}) {
  return (
    <div className="my-6 overflow-x-auto rounded-lg border bg-fd-card text-fd-card-foreground">
      <ol className="grid min-w-[760px] grid-cols-4 gap-3 p-4">
        {steps.map((step, index) => (
          <li
            className="relative rounded-md border bg-fd-background p-3"
            key={`${step.title}-${index}`}
          >
            <div className="mb-2 flex items-center gap-2">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-full border bg-fd-background text-[11px] font-medium text-fd-muted-foreground">
                {index + 1}
              </span>
              {step.role ? (
                <span className="rounded-sm border px-1.5 py-0.5 text-xs text-fd-muted-foreground">
                  {step.role}
                </span>
              ) : null}
            </div>
            <h3 className="m-0 text-sm font-semibold">{step.title}</h3>
            {step.detail ? (
              <p className="mt-1 text-xs leading-5 text-fd-muted-foreground">
                {step.detail}
              </p>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}
