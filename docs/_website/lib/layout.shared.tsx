import { DocsLayoutProps } from "fumadocs-ui/layouts/docs";
import { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { BookOpenText, GitBranch, Library } from "lucide-react";

/**
 * Shared layout configurations
 *
 * you can configure layouts individually from:
 * Home Layout: app/(home)/layout.tsx
 * Docs Layout: app/docs/layout.tsx
 */
export const baseOptions: Partial<DocsLayoutProps> & BaseLayoutProps = {
  nav: {
    title: (
      <div className="flex gap-4 items-center">
        <img src="/favicon.png" className="w-6 object-contain"></img>
        Tatva
      </div>
    ),
  },
  i18n: false,
  links: [
    {
      type: "icon",
      url: "https://github.com/smec-ethz/tatva",
      icon: <GitBranch />,
      text: "Github",
      on: "all",
    },
  ],
  sidebar: {
    tabs: [
      {
        title: "Docs",
        description: "Guides and tutorials",
        icon: <BookOpenText size={24} className="text-[var(--color-route-docs)]" />,
        url: "/docs",
      },
      {
        title: "API",
        description: "API reference",
        url: "/api",
        icon: <Library size={24} className="text-[var(--color-route-api)]" />,
      },
    ],
  },
};
