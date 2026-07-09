import Link from 'next/link';
import { SiteNav } from '@/components/site-nav';
import { SiteSearchTrigger } from '@/components/site-search-trigger';
import { TeaserImage } from '@/components/teaser-image';
import { WorldFoundryWordmark, WorldFoundryWordmarkLink } from '@/components/worldfoundry-wordmark';
import { withBasePath } from '@/lib/site-path';

const videos = [
  { src: '/home-demos/grid_00.mp4', label: 'Cosmos 3 Nano' },
  { src: '/home-demos/grid_01.mp4', label: 'Longcat' },
  { src: '/home-demos/grid_02.mp4', label: 'Cosmos 3 Nano' },
  { src: '/home-demos/grid_03.mp4', label: 'CogVideoX' },
  { src: '/home-demos/grid_04.mp4', label: 'Astra' },
  { src: '/home-demos/grid_05.mp4', label: 'Astra' },
  { src: '/home-demos/grid_06.mp4', label: 'Warp' },
  { src: '/home-demos/grid_07.mp4', label: 'AC3D' },
  { src: '/home-demos/grid_08.mp4', label: 'Warp' },
  { src: '/home-demos/grid_09.mp4', label: 'Cosmos 3 Nano' },
  { src: '/home-demos/grid_10.mp4', label: 'CogVideoX' },
  { src: '/home-demos/grid_11.mp4', label: 'Cosmos 3 Nano' },
  { src: '/home-demos/grid_12.mp4', label: 'Astra' },
  { src: '/home-demos/grid_13.mp4', label: 'CogVideoX' },
  { src: '/home-demos/grid_14.mp4', label: 'Warp' },
  { src: '/home-demos/grid_15.mp4', label: 'Astra' },
];

export default function HomePage() {
  return (
    <main className="pi-home-shell" style={{ maxWidth: '100vw', overflowX: 'hidden' }}>
      <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8 md:py-12">
        <header className="pi-header !mb-12 border-b border-[var(--pi-line)] pb-4">
          <div className="flex flex-wrap items-center justify-between w-full">
            <WorldFoundryWordmarkLink variant="header" />
            <div className="pi-site-header-tools ml-auto">
              <SiteNav active="home" />
              <SiteSearchTrigger />
            </div>
          </div>
        </header>

        <div className="flex flex-col items-center text-center mt-8">
          <WorldFoundryWordmark as="h1" variant="hero" />
          <p className="mb-12 max-w-3xl text-left text-lg leading-relaxed text-[var(--pi-muted)] indent-8">
            A Unified Codebase and Taxonomy for World Models. We formulate world modeling as faithful simulation of the real physical world rather than mere reproduction of observed appearances, introducing a principled capability taxonomy spanning perception, manipulation, dynamics, and interaction.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 mb-16 text-sm">
            <a href="https://openenvision.github.io/WorldFoundry" target="_blank" rel="noreferrer" className="flex items-center gap-2 border border-[var(--pi-line)] bg-white px-4 py-2 font-bold text-[var(--pi-ink)] transition-colors hover:border-[var(--pi-ink)] hover:bg-[var(--pi-paper)]">
              <span>Project Page</span>
            </a>
            <a href="https://github.com/OpenEnvision/WorldFoundry" target="_blank" rel="noreferrer" className="flex items-center gap-2 border border-[var(--pi-line)] bg-[var(--pi-ink)] text-white px-4 py-2 font-bold transition-opacity hover:opacity-90">
              <span>Code Repository</span>
            </a>
          </div>

          <TeaserImage
            variant="home"
            alt="WorldFoundry: a unified infrastructure and large-scale arena for world models"
          />
        </div>

        <section className="mb-24">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="mb-6 font-serif text-3xl text-[var(--pi-ink)]">Beyond Perceptual Realism</h2>
            <div className="text-left text-base leading-relaxed text-[var(--pi-muted)] space-y-4 indent-8">
              <p>
                Recent advances in generative modeling across video, 3D, and emerging 4D modalities have fueled the view that these systems are evolving into world models capable of simulating dynamic environments. Yet existing work often conflates perceptual realism with genuine simulation ability.
              </p>
              <p>
                Perceptual realism, however impressive, is neither necessary nor sufficient for genuine simulation competence. WorldFoundry clarifies this boundary, separating perceptual distribution modeling from stateful, action-conditioned simulation.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-24">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="mb-6 font-serif text-3xl text-[var(--pi-ink)]">Unified Taxonomy and Operational Definition</h2>
            <div className="text-left text-base leading-relaxed text-[var(--pi-muted)] space-y-4 indent-8">
              <p>
                A fundamental obstacle to progress in world modeling research is the absence of a precise and shared conceptual foundation. We formalize world modeling as the objective of simulating environment dynamics rather than merely reproducing observations, and introduce a capability gradient that traces the progression from <strong>perception</strong> and <strong>representation</strong> to <strong>manipulation</strong>, <strong>dynamics</strong>, and <strong>interaction</strong>.
              </p>
              <p>
                This taxonomy clarifies the boundary between foundation generative models and generative world models. A generative world model must support three tightly coupled operations: estimating latent state from observations, predicting future states under physical constraints, and coherently adapting futures under agent-driven interventions.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-24 bg-white/50 border border-[var(--pi-line)] p-8 md:p-12 shadow-sm rounded-2xl">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="mb-6 font-serif text-3xl text-[var(--pi-ink)]">A Unified Training, Inference and Evaluation Framework</h2>
            <div className="text-left text-base leading-relaxed text-[var(--pi-muted)] space-y-4 indent-8">
              <p>
                Contemporary systems span a broad landscape, yet the infrastructure used to study them remains fragmented. WorldFoundry provides a shared abstraction spanning model development, deployment, and assessment. It serves as a full-stack open-source framework covering data construction, training, unified inference across multimodal observations, and interactive visual analytics.
              </p>
              <p>
                Through Artifact and Representation Contracts, WorldFoundry standardizes how heterogeneous models expose their predictions and world-state estimates, enabling cross-family comparison without collapsing systems into an undifferentiated benchmark wrapper.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-20">
          <div className="mb-8 flex items-baseline justify-between border-b border-[var(--pi-line)] pb-4">
            <h2 className="font-serif text-3xl text-[var(--pi-ink)]">Showcase Demos</h2>
            <Link href="/docs/guides/supported-models" className="text-sm font-bold uppercase text-[var(--pi-muted)] hover:text-[var(--pi-ink)] transition-colors">
              View All Models →
            </Link>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 gap-4">
            {videos.map((video, idx) => (
              <div key={idx} className="group flex flex-col bg-white border border-[var(--pi-line)] p-2 transition-all hover:-translate-y-1 hover:border-[var(--pi-ink)] hover:shadow-[4px_4px_0_var(--pi-ink)]">
                <div className="relative aspect-video w-full overflow-hidden bg-black/5 border border-[var(--pi-line)]">
                  <video
                    src={withBasePath(video.src)}
                    autoPlay
                    loop
                    muted
                    playsInline
                    preload="metadata"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="mt-3 flex items-center justify-between px-1">
                  <span className="text-xs font-bold text-[var(--pi-ink)] truncate max-w-[70%]">{video.label}</span>
                  <span className="text-[9px] uppercase tracking-wider text-[var(--pi-muted)]">Generated</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-3xl text-center">
          <p className="pi-note">
            The framework is designed for contributors who need repeatable inference demos,
            reviewable metrics, and simple extension points. Start with the{' '}
            <Link href="/docs/quickstart">quickstart</Link>, validate inference outputs, then run
            evaluation or training only when the selected model family supports it.
          </p>
        </section>
      </div>
    </main>
  );
}
