// renovate config
Object.assign(process.env, {
  GIT_AUTHOR_NAME: 'Renovate Bot',
  GIT_AUTHOR_EMAIL: 'buildbot@act3-ace.com',
  GIT_COMMITTER_NAME: 'Renovate Bot',
  GIT_COMMITTER_EMAIL: 'buildbot@act3-ace.com',
});

module.exports = {
  endpoint: process.env.CI_API_V4_URL,
  hostRules: [
    {
      matchHost: 'https://reg.github.com/act3-ace/',
      username: process.env.CI_REGISTRY_USER,
      password: process.env.CI_REGISTRY_PASSWORD,
    },
  ],
  platform: 'gitlab',
  username: 'renovate-bot',
  autodiscover: true,
  automerge: true,
};