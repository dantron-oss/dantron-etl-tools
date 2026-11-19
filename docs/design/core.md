# Core: the tool contract, the boundary, the registry

Short notes on decisions in the core modules (`contract.py`, `boundary.py`, `ports.py`,
`status.py`, `config.py`, `frame_checks.py`, `registry.py`, `conformance.py`, `errors.py` and their
neighbours). Each heading is one decision: what was decided, why, and what was rejected.

## The tool protocol

**Decided.** A tool is any class that has the members of the `Tool` protocol. It is a
`typing.Protocol` marked `runtime_checkable`, so the registry can check the members are present when
a tool is registered.

**Why.** The registry only needs "a described thing that can be called". Structural typing says
exactly that without making a tool author inherit from one of our classes.

**Rejected.** An abstract base class. It would force inheritance and say nothing more.

### Two validation methods

`validate_config` sees the settings alone, so it holds cross-field rules. `validate_inputs` also
sees the frames, so it holds rules such as "the column this setting names must be on this port".

Both are part of the protocol and both are called on every run. An earlier shape called
`validate_inputs` only where a tool happened to define it. That made the check a rule that held
only while an author remembered it. The price of the stricter shape is that a tool with nothing to
check must still write `return None`, and a tool without the method is not a `Tool` at all.

`validate_config` must not read a file, a database or an environment variable. The loader calls
it, and a flow must load the same way whatever is on disk, so the viewer and the dead-end check can
read it. A check on a file belongs in `validate_inputs`, which runs only when the step runs.

## A fixed set of categories

**Decided.** `CATEGORIES` is a published, checked set: `input`, `transform`, `combine`,
`preparation`, `text`, `output`, `assert`.

**Why.** The catalog is assembled tool by tool from whatever each author typed. Left free, one group
becomes "transform", "Transform", "transformation" and "" on the rendered page. The same reasoning
applies to the published column type names.

**Terminal categories.** `output` and `assert` steps have their effect outside their ports: an
output writes somewhere, an assert raises or does not. So a step of either category may end a flow
without being a dead end. This follows the tool's own category, not a flag on the step, because a
per-step flag is one an author could set and forget.

## Worked examples are run

**Decided.** `check_tool` runs every `WorkedExample` through `run_tool` and compares the result with
the stated `outputs`. A documented example is a checked example. There is no exemption: a tool that
needs a database writes its example against in-memory SQLite.

### Row order is off by default

`check_row_order` defaults to `False`. For a tool that pairs or groups rows, the order rows come back
in is up to the query engine. Pinning it would make an ordinary example fail when the engine's
internals changed, not when the tool did. An example for an order-preserving tool opts in, and the
setting applies to every port the example documents.

### `setup` is a function that returns a context manager

An example that needs a world (a connection, a file already on disk) declares it through `setup`.
The rejected alternative was a tool building that world inside `help()`, as a side effect of
describing itself.

`setup` is a callable, not a context manager. The same example is checked more than once in one
process: once on its own, and again each time its catalog is checked. A context manager built once
and stored, such as a `@contextmanager` generator object, works for exactly one `with` block and is
exhausted the second time. A callable gives a fresh one on every call.

### The `{scratch}` directory

When an example's configuration holds `{scratch}`, `check_tool` opens a temporary directory before
`setup` is entered and replaces the token in a private copy of the configuration. `setup` reaches
the same directory through `scratch_path`. The directory is removed after the check, whatever the
check decided.

The directory belongs to the checker, not the example. Anything written inside it goes with it. A
file an example writes anywhere else is not seen or cleaned up by any check; this does not close
that hole.

## Input ports from the settings

**Decided.** A tool whose input ports depend on its settings (one port per sheet in `write_excel`)
leaves `input_ports` empty and implements `input_ports_for(config)` (the `DerivesInputPorts`
protocol). Every reader of ports asks `contract.input_ports_for`, so no two parts of the library get
different answers.

Three rules for the author:

- **One answer, never two.** A tool that declares `input_ports` and also implements the member gives
  two answers to "what does this step read". Different parts of the library would ask different
  ones, so a frame could be accepted by one and thrown away by the other with nothing raised.
  `check_tool` refuses a tool that does both.
- **Never name a port twice.** A port holds one frame, so the second use of a name would replace the
  first and lose a frame silently. `check_tool` refuses a worked example that does it, and the
  loader refuses a step that does it.
- **The settings arrive as written, with no defaults filled in.** The loader checks a step's edges
  before it validates the settings, and the boundary checks the ports before it fills defaults. So
  the field naming the ports must be one the author always writes. Nothing checks this rule, because
  an absent field and an empty one look the same from inside the member.

The catalog has no step's settings, so it shows the ports of the tool's first worked example
(`catalog_input_ports`). Those names cannot drift, because `check_tool` runs that example. They are
true of that one example only, which is why the catalog shows `ports_from_config` beside them.

## Errors from a tool's own code

`call_tool_code` wraps every call into code a tool author wrote: building the template, validating,
running, describing itself. The query engine can raise through any of these with an exception type
the caller has never heard of, so anything that is not a `ToolError` becomes an `ExecutionError`
naming the tool and the phase. A `ToolError` passes through first and unchanged, and this happens
at every phase, not only around `execute`: a tool that raises a precise `ConfigError` from its own
`validate_config` must reach the caller as itself.

The phase words (`_PHASE_VERBS`) are written once so that every call site names a phase the same
way and a reader can search a log for them.
