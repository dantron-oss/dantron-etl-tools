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

## One set of type names

**Decided.** Every tool that lets a flow state a column's type uses the names in `column_types.py`,
spelled as polars spells them: `String`, `Int64`, `Float64`, `Boolean`, `Date`, `Datetime`, `Time`.

**Why.** Several tools need these words. A CSV states no types, so its reader infers them and must
be correctable. A cast step states them outright, because two frames about to be stacked must agree.
Tools meant to answer a shared question the same way drift apart once the answer is written in two
files. The polars spelling means a reader who knows one knows the other, and none of the names is a
word YAML would turn into a boolean, a null or a number.

Only `Date`, `Datetime` and `Time` take a format when read from text. A source that already holds
real dates needs none, which is why not every tool offers one.

**A fraction is never cut silently.** polars reads 1.99 as the whole number 1 and -2.7 as -2,
without a word. `cut_fractions` finds those rows first, and a tool refuses with `fraction_refusal`
(the owner's ruling: refuse unless every value is already whole, and say to `round` first).

## One name, one tool

`ToolRegistry.register` refuses a second tool with a name already taken. Letting it overwrite the
first would be a defect a caller finds only by noticing the wrong tool ran. `names()` is sorted so
the rendered catalog does not reshuffle as tools are added.

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

**Why a token.** `setup` builds its world too late for the example's own configuration to see it.
Before the token, a tool that touched disk had a fixed path frozen into its example, and the
catalog showed readers a path such as `/tmp/etl_worked_example_figures.xlsx`. The token is replaced
only in a private copy of the configuration; the stored example, which the catalog page reads,
still shows `{scratch}`. That is also why the replacement rebuilds lists and dicts instead of
writing into them: writing in place would resolve the token for whichever check ran first and
leave the page showing a directory that existed for a few milliseconds.

The replacement uses a function, not a replacement string, because `re.sub` reads backslashes in a
string as escapes and every Windows path is full of them.

The directory is held in a `ContextVar`, not a module global, so a `setup` sees exactly the
directory its own check opened and never one left over from an earlier check.

### Why `${scratch}` is left alone

`{scratch}` is one token with one meaning, not the start of a substitution language. A label such
as `{total}` reaches a tool exactly as written.

`${scratch}` is a different thing: an unbound rule reference, which the placeholder check must
refuse. The pattern refuses to match a `{scratch}` that follows `$`. Without that, the token would
be replaced before the placeholder check ran, turning a configuration nobody bound into one that
runs, against a directory about to be removed.

### The disturbance check

After describing a tool and after each example, the checker compares the environment variables and
the working directory with a snapshot. These two are the realistic disturbances that can be seen
cheaply from outside a tool's code. A file left on disk, an open connection, a module global or a
changed `sys.path` are not seen, and the check does not claim to see them.

`temporary_environment` puts back what was there: a variable that had a value gets it back, never
removed, because the process doing the checking keeps running afterwards. A variable that was
absent is removed again. Both hold whether the block finishes or raises. Without the helper, each
reader and writer tool would hand-write this pair and get one half wrong.

## A worked example database

**Decided.** A tool that reads or writes a database writes its example against a shared in-memory
SQLite database, opened by `example_database` inside the example's `setup`.

**Why.** `help()` must describe a tool without changing the process that asks. The catalog page
generator calls `help()` on every tool, and `check_tool` refuses a description that opens a
connection or sets a variable.

The seeding connection stays open for the whole block, because a shared in-memory database lives
only while a connection to it is open. It is closed on the way out in every case, so the same
example can run again the next time the catalog is checked.

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

## One error family

**Decided.** Every error the library raises descends from `ToolError`, so a script that drives a
flow has one type to catch. The subclasses say what to look at next: a port (`PortError`), a
step's settings (`ConfigError`), the wiring (`PipelineError`), the catalog (`ConformanceError`,
`RegistryError`), the data (`DataError`), the run (`RunError`), or something nobody expected
(`ExecutionError`).

`ToolError` is a `ValueError`, so a caller that already treated configuration problems as
`ValueError` keeps working without knowing the specific types exist.

`PipelineError` and `ConfigError` are kept apart because they send the reader to different places:
the first to the steps and arrows, the second to one step's settings block.

## Unexpected failures are wrapped

**Decided.** The boundary turns any exception from a tool's own code that is not a `ToolError` into
an `ExecutionError`, with the original as `__cause__`.

**Why.** Every tool reaches into polars somewhere, and polars raises its own types (`SchemaError`,
`InvalidOperationError`). Asking each tool to catch and translate its own calls is a convention,
and the first tool that forgets breaks the promise of one family to catch. So the boundary
translates.

**The cost, stated openly.** An `ExecutionError` may wrap a failure that has nothing to do with the
tool: memory the machine could not give, a file that could not be read. The family does not tell
that case apart from a bug in the tool.

**Rejected.** A hand-kept list of "not really the tool's fault" exception types. It would always
have a gap nobody noticed until it mattered. A driver that wants to carry on past one of these
decides by looking at `__cause__`, never by having caught `ExecutionError`.

## A data error keeps counts and examples

**Decided.** `DataError` means the rows themselves were wrong. It is not a kind of `ExecutionError`,
in either direction, and the boundary passes it through unwrapped.

**Why.** The reader's next move differs. A data error says "look at the data". An execution error
says "look at the run", and may be wrapping something the machine did. Wrapped, a data error would
announce that a tool "failed while running" and hide the one message that mattered.

The constructor reduces the offending rows to two counts and a few example rows, and does not keep
the frame:

- The counts and examples make the refusal useful to someone who was not there. "3 of 5 rows,
  starting with ..." sends them to the rows. "An invariant failed" sends them to write a query
  first.
- Keeping the frame would hold it in memory for as long as the exception is reachable, which for an
  exception leaving a long flow is the rest of the run.

Only `MAX_REPORTED_ROWS` (5) example rows are printed, because the message is read later in a log
and a check that fails on every row must not paste the whole frame into it. The counts are never
capped. When the rows were refused as a whole and never examined one by one, the message says so
instead of "0 of N", which would read as "no row was at fault".

`name_choices` lists at most `MAX_LISTED_NAMES` (10) names, because a flow can have a thousand steps
and a message listing them all is a page nobody reads.

## Named ports

**Decided.** Every input and output of a tool travels in a `Ports`: named frames, fixed once built.
`run_tool` refuses, by name, a declared port the tool did not produce, an output nobody declared, an
output with no columns, a result that is not `Ports`, and a change to an input.

**Why.** A step may have more than one output: the rows that matched and the rows that did not. A
side output a caller forgets to read should not show up three steps later as a bare `KeyError`. An
output with no columns would make a later "this column must exist" check fail for the wrong reason,
so an output must keep its columns even with no rows.

**Rejected.** A plain `dict` of frames in and out.

### `Ports` is a closed, read-only class

`Ports` does not inherit from `collections.abc.Mapping`. `Mapping.get` works by catching `KeyError`
from `__getitem__`, but a missing port raises `PortError` here. Inheriting the mixin would give a
`get` that is broken in a way nobody would notice. A small, closed set of methods is easier to
trust than a partly working base class.

The source mapping is copied when a `Ports` is built. `__slots__` and read-only properties keep it
unchanged afterwards: there is nowhere for a new attribute to go, and no property has a setter.

## Notes and per-row status

A step can tell the reader something short of refusing, in two ways that travel together in
`Ports`, beside the frames.

**Notes** say how many: "595 values in 'amount' did not read as Float64 and are now empty". A tool
that quietly emptied three values out of forty-six thousand has done something a reader wants to
know the size of, but an exception would stop the run. The runner turns a step with notes into a
warning, which puts it in the log and marks it on the canvas. A note is for a person; nothing
downstream may read one.

**Per-row status** says which rows. After a failed cast every one of those values is null, and a
null that failed to read looks like a null that was always there. Given only the count, a reader
cannot tell a figure the run destroyed from text that was never a figure. The status frame has one
row per data row, in the same order, and is never worked out again afterwards.

### Beside the frame, never inside it

**Rejected.** A `status` column in the frame. A column flows into the next join, survives a
`select_columns` that never heard of it, turns up in a schema comparison, and lands in the published
workbook. It would have to be carried and stripped at every one of those places, and the first
place that forgot would ship it. What a step did is a fact about the step, not the data, so it
travels where notes travel.

Almost every step reports no status. The presence of a status on a port is itself the signal, so
the boundary refuses one that says every row is fine.

### The shape of a status frame

One published shape, `STATUS_SCHEMA`: a `level` and a `reason`.

- `level` is one of `ok`, `warning`, `error` (`STATUS_LEVELS`). The reader of a status is a page
  renderer. A level it has no drawing for shows as nothing at all, so the row reads as fine. Hence
  three words only, and the boundary refuses a fourth.
- `reason` is a list, because one row can be changed twice in a step (two columns cast, both
  unreadable on that row). One reason per row would drop the second silently and send a reader to
  the wrong column.
- Each reason is a `{column, detail}` struct, not a sentence. The question asked of a status is
  "which rows does this warning cover?", answered by matching `reason.column` exactly. Against
  prose that would be a substring search.
- A reason quotes at most `STATUS_VALUE_CHARS` (60) characters of the value. A reason is read in a
  table cell, so it is a caption, not a copy. Sixty characters was enough to recognise every value
  the run that prompted this had emptied, and the full value is on the input port. An uncapped
  quote on a column that failed six hundred times would copy the frame into the status.

`value_caption` never returns null. A reason built with `polars.format` around a null becomes null
as a whole, and a marked row whose reason vanished cannot say anything. A list or struct column has
no text form and is refused by polars; in practice a column is only captioned straight after a cast
to a single-value type, which refuses first.

### One helper for unreadable values

`unreadable_report` is shared by the two tools that read a column as a stated type, so they say it
in the same words. It returns the reasons and the note from one call, both read off the same mask,
so the count in the note can never disagree with the rows that are marked. The mask is the step's
own, taken before the values were replaced: afterwards a value that failed and a value that was
always null look the same. A blank gets no reason, because it never failed.

The note quotes the first offending value. "595 values failed" sends a reader looking; "595 values
like 'Acme Ltd' failed" usually settles it on the spot.

The share is printed with two decimal places. The general format `.2g` printed 100% as `1e+02`, and
100% is the share a text column reaches most often.

### Row status checks each mark

`row_status` checks the length of every mark against the port's height. The boundary only compares
a finished status with its port, which catches a wrong total and nothing finer. A tool that stacks
one block per melted column could hand over one block a row short and another a row long. The total
would match, and every mark between them would sit on the wrong row.

A mark is one series of reasons (null where nothing happened), not a mask beside a sentence: two
separate things could disagree, leaving a row marked and unexplained. Marks are applied in the order
given, which is the order the step did the work. With no marks `row_status` returns all `ok`. That
is not a status to publish, but a tool stacking one status per block needs it for clean blocks.

## The boundary

`run_tool` is the one place every call to a tool goes through. It calls into the tool's own code
five times, in a fixed order:

1. `input_ports_for`: which ports the step reads, from the settings as written (the same settings
   the loader asks about), then refuse a missing or extra port.
2. `config_template`, then the template check and default filling.
3. `validate_config`, on the filled settings.
4. `validate_inputs`, on the filled settings and the frames.
5. `execute`.

**Ports first**, because a port mistake belongs to the flow that wired the step, not to anybody's
settings. **The template is checked here, not by each tool**: a tool with nothing beyond its
template naturally writes an empty `validate_config`, and that once bought it no validation at all.
**`validate_config` runs on the filled settings**, so a cross-field rule can read a defaulted value
with `config["x"]` instead of guessing with `.get`. `execute` never sees settings that fail
validation.

**`validate_inputs`** is the only phase that sees the settings and the frames together. Neither a
template nor a cross-field rule has a frame, so "does the column this setting names exist on the
port it will be read from?" used to live in the first lines of each tool's `execute`, in each tool's
own words, after the boundary had already reported the tool as running.

A result that is not `Ports` is refused by name. Returning a bare dict of frames is the likeliest
slip, and unchecked it shows up as an attribute error inside `run_tool` naming neither the tool nor
the mistake.

The settings are deep-copied before use, so a tool that appends to a list in its settings cannot
change the step it was given, or that step's fingerprint.

The outputs are rebuilt in the declared order, carrying the notes and status across. Once the notes
were dropped by this very rebuild: a tool reported something and the boundary discarded it.

### Tools get a copy of their inputs

The tool receives a polars `clone` of each input frame. A clone shares the data and costs almost
nothing (about 0.01 ms on a million rows). A change the tool makes in place, such as rows appended,
a value set or a column added, lands on the copy. It never reaches the caller's frame, another
branch reading the same port, or a step's record of what it produced.

A change to the copy's columns or row count is still refused afterwards, because the author should
hear about it. A value changed in place is not looked for: it cannot reach anything but this tool's
own result, and comparing every value would cost a pass over every input on every step.

### What the boundary refuses in a status

Every status is checked, because a status row lines up with a data row by position and nothing
works it out again later. A status one row short does not report less; it reports the wrong rows,
sliding every mark after the gap onto its neighbour. Both directions of a height mismatch are
refused, since a check written for the short case alone would pass a status built per input row
before a filter.

Also refused: a status on a port that is not an output, the wrong shape, a level outside
`STATUS_LEVELS`, and, last, a status where every row is `ok`. Seven million rows saying nothing
happened would make the presence of a status mean nothing. This matches the library's ruling on a
cast to the type a column already holds: a declaration that does nothing is refused.

## The template is checked at the boundary

**Decided.** `run_tool` checks every step's settings against the tool's template. Tools do not do
this themselves. A tool with no cross-field rules has nothing of its own to write, and a rule that
holds only when an author remembers it is a convention, not a rule.

### Expression fields

A field of type `expression` is the link to the expression language. "Expression" is a field type,
not a separate flag on the field, so a field cannot be an expression in one place and a plain string
in another, and `expression_fields` has one thing to read.

`expression_fields` gives paths without indexes (`rules[].when`), because a template describes what
may be there and has no elements. That is what a form or a documentation page asks.
`expression_values` gives paths with indexes (`rules[0].when`), read from real settings. Only an
indexed path can be shown to a person who has to go and fix the expression. The loader lifts the
column names out of each one and checks them against the upstream frame before any row is read.

### Nested fields

`items` (an array's element) and `fields` (an object's members) let a template describe what is
inside a setting. The most common settings shape is an ordered list of small objects: rules,
aggregations, key pairs. Without these two, such a setting is one untyped `array`, and nothing inside
it is type-checked, required-checked or compiled, so its expressions are invisible to the loader.

### `required` and `non_empty`

They answer two questions: did a value arrive at all, and does it hold anything. `required` refuses
a missing value or `None`, and accepts `[]`, `""` and `{}`, each of which is a value. `non_empty`
refuses those empty values, with or without `required`. This keeps one meaning per word.

`non_empty` only applies to strings, expressions, arrays and objects. Zero is a full value, so
`non_empty` on a number would constrain nothing.

### Booleans are not numbers

`bool` is a subclass of `int` in Python, so a naive check accepts `true` where a template asks for a
count. Both numeric types refuse a boolean, matching the expression language's own whole-number
arguments.

### Every field is filled

`apply_defaults` puts every template field in the settings: its default, or `None`. Filling only
fields with a default would make `execute` read its own settings two ways, subscript for one field
and `.get` for the next, a distinction every author would rediscover field by field. With every
field present, "was this set?" is always `config["x"] is None`.

This works only because the template check accepts `None` for an optional field: filled settings
are validated again by the tool's own `validate_config` and must pass.

Nothing is filled inside an array or an object. There is no element to fill a member of, and
inventing one would add an entry to an ordered list nobody wrote.

`apply_defaults` is public because a loader building settings from a file wants the same filling.

### What the template check refuses

`validate_against_template` refuses an unknown field, a missing required one, the wrong type, an
empty value on a `non_empty` field, a value outside `enum`, and an expression that does not compile.
The expression is compiled with no known columns: bad syntax is caught here, while a missing column
is left to the caller, which knows the frame.

A `${...}` placeholder left in any string, at any depth and whatever the field's type, is refused.
Resolving it is someone else's job, but an unresolved placeholder would otherwise pass as ordinary
text, match nothing, and leave the run looking successful.

A field whose declared type is unknown is refused, not skipped. The lookup once failed open, so a
template author who guessed the name of a type that does not exist got a field that accepted
anything. `FIELD_TYPES` is read from the type checks themselves, so the published set and the
enforced set cannot differ.

### A key YAML turned into `True`

A settings key that is not a string gets its own refusal, separate from "unknown field". It comes
from a pipeline file: the author wrote `on: period`, YAML turned the bare word into `True`, and the
key stopped being a name. An unknown field is a typo, fixed with a different word. This is a name
the file changed, fixed with quotes. The message gives the type as well as the key, because `True`
and `"True"` print the same.

## Frame checks

The helpers in `frame_checks.py` each answer a question that needs both the settings and the frame
that arrived, so none can live in a template and none should wait for `execute`. Each exists
because several tools were already asking it, and the third or fourth spelling of a question is
where the answers start to differ.

### Expression errors become config errors

`compile_for` passes the frame's schema (names and types), not only its column names, so a
misspelled column and a function given the wrong type both fail here, in the author's words, instead
of one of them failing later inside the query engine.

It turns `ExpressionError` into `ConfigError`. Callers catch `ToolError`; a bare `ExpressionError`
would slip past them. `check_tool`'s example runner is such a caller, so a worked example with a bad
expression would otherwise escape the conformance check.

The refusal names the field and quotes the text, because it is read later in a log by someone
holding a file of many expressions.

`compiled_dtype` works from the schema, so it gives the same answer on a million rows and on none.
A tool that refuses a run because of it must not behave differently when a frame arrives empty. It
returns the column name with the type because every caller names the column in its refusal.

`compile_predicate_for` refuses an expression that does not give true/false. `[amount]` written for
`[amount] > 0` compiles, because that is how a column is written, and would otherwise fail inside
the query engine with a type complaint naming neither the field nor the text. A column that already
holds true/false is a valid condition as written: the test is what the expression gives, not what
it looks like.

### `require_columns` names the port

Every tool that names columns asks this, and each private version said it differently. The port is
why one shared version is worth more than several: "column not found" without saying `left` or
`right` sends the reader to both. The refusal lists the columns that are present, because the usual
mistakes are a misspelling and a wrong case, and both are obvious next to the real list.

### Division by zero stops the step

By the owner's ruling, a division by zero stops the step (`refuse_rows_without_an_answer`). Left to the
engine it gives infinity, or not-a-number for 0 / 0, and a later filter reads not-a-number as larger
than every number. The refusal names the field, the expression, the divisor and up to five rows. A
flow that expects zeros guards with `case_when` first; `case_when` passes `among` so a case that
sends zero rows elsewhere protects the case that divides.

A null divisor gives null, as everything does with a null, and is allowed. Not-a-number is not zero
either. The check is one whole-column comparison per `/`, with no row-by-row work.

The same function also stops the step on a row a function has no answer for. A function says which
rows those are in its entry in the function table (`no_answer`): text `parse_date` cannot read, a
count `add_days` cannot move by exactly. Every tool that reads an expression already calls this
function, so a new function with such a rule is checked in every tool, and inside `case_when` only
on the rows that reach the case. Its name, `refuse_rows_without_an_answer`, says it covers both.

### Writing a column name

`check_output_column` holds three rules for every tool that writes a column.

**A blank name is refused first.** A column called `"   "` is invisible on a page, looks like its
neighbours in a header row, and cannot be written in an expression. Before this check, seven tools
were asked and gave three different answers. Nothing is trimmed: `" total "` is a strange name, but
still a name. The blank is refused before `replace` is read: reading the flag first once gave one
mistake two explanations, and under `replace: true` a blank was reported as a column the input does
not have, which sends the author to the wrong file. The blank is not quoted back, since an empty
string and three spaces print the same.

A field that declares `non_empty` is refused earlier, at load, with a shorter message ("must not be
empty"), so only whitespace reaches this check on such a field. The two wordings are kept: the
load-time refusal is better because it needs no frame. This check is the backstop for fields whose
author did not declare it.

**Replacing a column must be asked for.** Refusing replacement outright is too strict, since
repairing a column in place is a real step. Taking the name silently is the hazard: whoever writes
one step does not know every column of the frame, and the overwritten value is gone. So a clash
nobody asked for is refused, naming the column, the field and the `replace` setting.

**`replace` on a missing column is refused too.** Unchecked, a misspelling quietly creates a new
column while the column meant for repair keeps its old values, and the run reports success.

`REPLACE_FIELD` is one shared definition so the wording is the same on every catalog page. It has no
default: a default would do nothing on a field inside an array element, so a missing value means
"not asked".

### Time zones

A zone-aware timestamp read as a date needs a stated zone (`ZONE_FIELD`, `require_stated_zone`), by
the owner's ruling, because which day a timestamp falls on depends on the zone. There is no default.

## What check_tool refuses

The structural rules (name, description, category, ports, template, help) run before any worked
example, since there is no point running an example for a tool without a name worth reporting. The
examples run last and matter most.

**Only an `output` tool may have no output ports.** A transform, combine or read that produces
nothing has been wired into a flow that can never use it. A sink producing nothing is the point of a
sink: a workbook writer has nothing honest to hand on, and a made-up receipt would be shape without
information.

**`input_ports` or `input_ports_for`, not both.** Doing both once meant the run guard accepted a
supplied frame while the skip decision read the static attribute; the step then ran and dropped the
frame silently. This was a line in a docstring before it was a check.

**A port named twice by an example's settings** is refused before any example runs. Ports are built
from a mapping, so a repeated name arrives as one frame and the boundary sees nothing wrong. The
example would then stack the same frame twice and be reported as drifted, sending the reader to
compare rows when the fault is the port list.

**`help()` must not change the process.** A snapshot of environment variables and working directory
is taken before `help()` and compared after. Describing a tool is not allowed to export a variable
or change directory. The same comparison runs around each example's `setup`.

### Template declarations that do nothing

A nested declaration is where a template is read least and trusted most. `check_tool` refuses any
declaration that looks as if it constrains something and does not: `items` outside an array,
`fields` outside an object, `non_empty` on a type with no notion of empty, a `default` below the top
level, and `required` on an array element.

A nested default is never applied (see "Every field is filled"), yet reads as if it were, leaving
`execute` to use `.get` on a field its template promised would be there. `required` on an element is
read by nothing, since an absent element is not an element; the author means `non_empty` on the
array, and the message says so. `required` on a member of an element is a different thing, is
enforced per element, and stays.

A declared default is checked against its own field here, and only here: validation runs on what a
caller supplied, and the default is added afterwards. Unchecked, a default is the one value that
never meets the validator.

### Names YAML would rewrite

YAML reads some bare words as other types: `true`, `no`, `on`, `off`, `null`, `~` and their
capitalised spellings become booleans or null, and numeric text becomes a number.
`YAML_UNSAFE_VALUES` lists them, published because a tool author choosing an option's name needs it.
`nan` and `inf` stay text in YAML, so the numeric check is a pattern, not a call to `float`.

**Field names** and **input port names** are keys in a pipeline file. A tool once had a field
called `on`, so the obvious way to write that step gave the loader the key `True`. Quoting the key
works, but it puts the fix on every author of every pipeline using the tool, with nothing telling
them. The tool has one author, so the refusal belongs in `check_tool`. Nested names are checked too.
Field names were once exempt on the argument that a bad one announces itself. It did not: a key of
another type crashed the code that reports an unknown field, and one level down it named a field
nobody had written.

**Output port names** are exempt: a step names one as a value, on the right of a colon, where YAML
leaves the word alone. `filter` has output ports called `true` and `false`. Ports worked out from a
step's settings are not checked yet.

**Options** (`enum` values) arriving as `True` are refused by the type check, which is confusing
but safe. An option arriving as `None`, which is what the obvious spellings of "nothing" become, is
read as a field nobody set and filled with the default, so the run carries on under a choice its
author did not make. Every such word has a safe synonym: `include`/`exclude`, `first`/`last`. The
numeric rule refuses options `"1"` and `"2"`, correctly: they would arrive as integers and never
match. The fix is a field type that is not a string.

### Every output port is compared

An example's settings are validated against the template and against the example's own inputs
before it runs. An example whose settings are wrong and an example that fails to run have different
fixes; reporting both as "does not run" leaves the reader to work out which. A documented example
is what people copy, so a settings fault should point at the line to change.

A `setup` stays open for validation as well as the run, because a source's `validate_config` is
where it resolves its connection. The teardown runs in `finally`: a failed check that left a
database or variable behind would affect every later check in the same process. A `ToolError`
passes through untouched; only a failure of the setup's own entry or exit is turned into a
`ConformanceError`, because that failure belongs to the setup.

Every declared output port is compared, not only the ones an example mentions. The unmentioned ones
are the side outputs (the rows that did not match, the leftovers from each side), which are the
likeliest to be wrong. An example documenting one of three outputs has been run, not checked.

`check_row_order` changes only the row order part of the comparison. Values, types, column order and
row count are always checked.

### A new scratch directory each time

The `{scratch}` directory is created before the settings are resolved and before `setup` is
entered, because a source resolves what it will read as early as `validate_config`. The settings
are resolved on a copy made with `dataclasses.replace`, so the example a reader copies still says
`{scratch}`.

Each call gets a new directory. The conformance tests run `check_tool` twice on purpose; a shared
directory would let a file a sink's example wrote on the first pass satisfy the second pass's
"nothing exists yet" without writing anything.

The directory is removed in `finally`. Whether it still existed is recorded first, and the
"destroyed the checker's directory" refusal is raised only after cleanup and only when nothing else
was raised, so a drifted example is still reported as drifted.

`check_registry` reports every failing tool at once, so a catalog can be fixed in one pass.

## The catalog lists every field property

**Decided.** `catalog()` builds each configuration field's entry by walking the `ConfigField`
dataclass, not by naming its properties one by one. It skips only a property that is `None`.

**Why.** A property added to `ConfigField` then reaches the catalog because it exists, not because
someone remembered a line in `catalog.py`. The hand-written version had three branches whose other
arm was "say nothing", which is a silent omission by construction. Skipping any false value would
have dropped a meaningful `default: false`.
