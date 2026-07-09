import { EcosystemPage } from '@/components/ecosystem-page';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Events',
  description: 'WorldFoundry meetings, demos, benchmark sprints, and release milestones.',
};

export default function EventsPage() {
  return (
    <EcosystemPage
      active="events"
      comingSoon="Meetings, demos, benchmark sprints, and release milestones will be posted here soon."
      description="Community meetups, demos, and project milestones."
      footerLabel="Events"
      label="Project calendar"
      title="Events"
    />
  );
}
