// UiB KIX key guard for OpenCode 2 (loaded by kix.json through its "plugins" list).
//
// opencode gets the KIX key from the KIX_API_KEY environment variable, which
// the opencode-kix launcher sets for opencode only. Every shell that opencode
// starts (the agent's shell tool and "!" commands) would normally inherit that
// variable. This plugin removes it from each shell's environment before the
// shell starts, so a command such as `env` or `printenv` cannot print the key.
//
// It changes nothing else: no network access, no files, no other variables.
// No import is needed: OpenCode 2 accepts a default export with an id and setup.
export default {
  id: "uib.kix.key-guard",
  async setup(ctx) {
    await ctx.shell.hook("create.before", (event) => {
      delete event.env.KIX_API_KEY
    })
  },
}
