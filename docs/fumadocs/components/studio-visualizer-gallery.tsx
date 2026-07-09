import { withBasePath } from '@/lib/site-path';

type StudioGalleryLocale = 'en' | 'zh';

type StudioVisualizerDemo = {
  id: string;
  title: Record<StudioGalleryLocale, string>;
  subtitle: Record<StudioGalleryLocale, string>;
  aliases: Record<StudioGalleryLocale, string>;
  image?: string;
  embedUrl?: string;
  demoUrl?: string;
  status?: Record<StudioGalleryLocale, string>;
  port: string;
  artifacts: string;
  command: string;
};

const labels = {
  en: {
    eyebrow: 'Studio visualizer demos',
    overviewCaption: 'WorldFoundry Workspace · Visualizers tab',
    overviewAlt: 'WorldFoundry Workspace visualizers overview',
    port: 'Port',
    aliases: 'Aliases',
    artifacts: 'Artifacts',
    launch: 'Launch route',
  },
  zh: {
    eyebrow: 'Studio 可视化 Demo',
    overviewCaption: 'WorldFoundry Workspace · Visualizers 页面',
    overviewAlt: 'WorldFoundry Workspace 可视化入口总览',
    port: '端口',
    aliases: 'Alias',
    artifacts: '产物',
    launch: '启动入口',
  },
} satisfies Record<StudioGalleryLocale, Record<string, string>>;

const demos = [
  {
    id: 'world',
    title: {
      en: 'World Rollout Viewer',
      zh: '世界模型分段 Rollout',
    },
    subtitle: {
      en: 'Buffered realtime target for action-conditioned roaming. Controls are sampled continuously while generated chunks are prefetched.',
      zh: '面向实时漫游的缓冲式动作条件 rollout。前端持续采样控制输入并预取生成片段。',
    },
    aliases: {
      en: 'interactive-world, world-model, world-rollout',
      zh: 'interactive-world, world-model, world-rollout',
    },
    image: '/images/studio/visualizers/live/interactive-world-model.png',
    port: '7868',
    artifacts: 'image, video, action_trace',
    command: 'python -m worldfoundry.studio.app matrix-game-2 --frontend world --port 7868',
  },
  {
    id: 'viser',
    title: {
      en: 'Viser Geometry Viewer',
      zh: 'Viser Geometry Viewer',
    },
    subtitle: {
      en: 'Point clouds, depth-derived point sets, meshes, camera poses, and trajectories.',
      zh: '点云、depth 转 point set、mesh、camera pose 和 trajectory。',
    },
    aliases: {
      en: 'viser, geometry, pointcloud',
      zh: 'viser, geometry, pointcloud',
    },
    image: '/images/studio/visualizers/live/viser-geometry.png',
    port: '18590',
    artifacts: 'ply, pcd, xyz, glb, gltf, obj, npz',
    command: 'python -m worldfoundry.studio.app pi3 --frontend points --asset /path/to/scene.ply',
  },
  {
    id: 'rerun',
    title: {
      en: 'Rerun Timeline Viewer',
      zh: 'Rerun Timeline Viewer',
    },
    subtitle: {
      en: '`.rrd` timelines and recordings with synchronized cameras, frames, tracks, and 3D entities.',
      zh: '`.rrd` timeline / recording，同步 camera、frame、track 与 3D entity。',
    },
    aliases: {
      en: 'rrd',
      zh: 'rrd',
    },
    image: '/images/studio/visualizers/live/rerun-timeline.png',
    port: '9876',
    artifacts: 'rrd',
    command: 'python -m worldfoundry.studio.app vggt --frontend rerun --asset /path/to/recording.rrd',
  },
  {
    id: 'spark',
    title: {
      en: 'Spark Gaussian Splat Viewer',
      zh: 'Spark Gaussian Splat Viewer',
    },
    subtitle: {
      en: '3D Gaussian splats via the in-tree Spark frontend. Live preview uses the Spark.js Hello World butterfly demo.',
      zh: '通过仓内 Spark frontend 查看 3D Gaussian splat。实时预览来自 Spark.js Hello World butterfly demo。',
    },
    aliases: {
      en: '3dgs, splat',
      zh: '3dgs, splat',
    },
    embedUrl: 'https://sparkjs.dev/examples/hello-world/',
    demoUrl: 'https://sparkjs.dev/examples/#hello-world',
    port: '8765',
    artifacts: 'splat, spz, ksplat, sog, splat-ply',
    command: 'python -m worldfoundry.studio.app vggt --frontend spark --asset /path/to/scene.splat',
  },
  {
    id: 'embodied',
    title: {
      en: 'Embodied Simulator Bridge',
      zh: '具身仿真器 Bridge',
    },
    status: {
      en: 'Planned',
      zh: '计划中',
    },
    subtitle: {
      en: 'Planned integration. Studio only registers an external simulator URL and tunnel hints today; there is no in-tree embodied viewer yet. Start the simulator separately, then paste its URL in Visualizers.',
      zh: '仍处于计划/早期集成阶段。当前 Studio 只会登记外部 simulator URL 并给出 tunnel 提示，还没有仓内 embodied viewer。请先单独启动 simulator，再把 URL 填到 Visualizers。',
    },
    aliases: {
      en: 'sim, simulator',
      zh: 'sim, simulator',
    },
    image: '/images/studio/visualizers/live/embodied-simulator.png',
    port: '18610',
    artifacts: 'action_trace, trajectory, simulator_url',
    command: 'python -m worldfoundry.studio.app openvla --frontend embodied --simulator-url http://127.0.0.1:18610',
  },
] satisfies StudioVisualizerDemo[];

export function StudioVisualizerGallery({ locale = 'en' }: { locale?: StudioGalleryLocale }) {
  const t = labels[locale];

  return (
    <section className="pi-studio-viz-gallery not-prose" aria-label={t.eyebrow}>
      <figure className="pi-studio-viz-overview">
        <div className="pi-studio-viz-overview-media">
          <img
            src={withBasePath('/images/studio/visualizers/live/workspace-overview.png')}
            alt={t.overviewAlt}
            className="pi-studio-viz-overview-image"
            loading="lazy"
          />
        </div>
        <figcaption className="pi-studio-viz-overview-caption">{t.overviewCaption}</figcaption>
      </figure>

      <div className="pi-studio-viz-grid">
        {demos.map((demo) => {
          const media = demo.embedUrl ? (
            <iframe
              src={demo.embedUrl}
              title={demo.title[locale]}
              className="pi-studio-viz-card-embed"
              loading="lazy"
              referrerPolicy="no-referrer"
              tabIndex={-1}
            />
          ) : (
            <img
              src={withBasePath(demo.image ?? '')}
              alt={demo.title[locale]}
              className="pi-studio-viz-card-image"
              loading="lazy"
            />
          );

          return (
          <article
            className={['pi-studio-viz-card', demo.status ? 'pi-studio-viz-card-planned' : '']
              .filter(Boolean)
              .join(' ')}
            key={demo.id}
          >
            <div className="pi-studio-viz-card-media">
              {demo.demoUrl ? (
                <a
                  href={demo.demoUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="pi-studio-viz-card-media-link"
                  aria-label={`${demo.title[locale]} — Spark.js demo`}
                >
                  {media}
                </a>
              ) : (
                media
              )}
            </div>
            <div className="pi-studio-viz-card-body">
              <div className="pi-studio-viz-card-head">
                <div className="pi-studio-viz-card-title-row">
                  <h3 className="pi-studio-viz-card-title">{demo.title[locale]}</h3>
                  {demo.status ? (
                    <span className="pi-studio-viz-card-status">{demo.status[locale]}</span>
                  ) : null}
                </div>
                <ul className="pi-studio-viz-card-facts">
                  <li>
                    <span className="pi-studio-viz-card-fact-label">{t.port}</span>
                    <code>{demo.port}</code>
                  </li>
                  <li>
                    <span className="pi-studio-viz-card-fact-label">{t.aliases}</span>
                    <code>{demo.aliases[locale]}</code>
                  </li>
                </ul>
              </div>
              <p className="pi-studio-viz-card-desc">{demo.subtitle[locale]}</p>
              <p className="pi-studio-viz-card-artifacts">
                <span className="pi-studio-viz-card-fact-label">{t.artifacts}</span>
                {demo.artifacts}
              </p>
              <div className="pi-studio-viz-card-cmd">
                <span className="pi-studio-viz-card-fact-label">{t.launch}</span>
                <code>{demo.command}</code>
              </div>
            </div>
          </article>
          );
        })}
      </div>
    </section>
  );
}
