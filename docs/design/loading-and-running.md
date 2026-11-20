# Loading and running a flow

Short notes on decisions in the modules that read a pipeline file and run it: `loader.py`,
`spec.py`, `rules.py`, `renames.py`, `migrate.py`, `walk.py`, `runner.py`, `trace.py`, `focus.py`
and `preview.py`. Each heading is one decision: what was decided, why, and what was rejected.

## A flow is a data file

A flow written as data says which tools are wired how. A person can read it and change it, and
changing it changes what a run does. A flow written as functions could only be edited by generating
functions. That is why `Pipeline` and `Step` hold what the file says and nothing more, and why
loading moves no rows.

## The order of the loader's checks

**Decided.** `load_pipeline` refuses everything it can find without a frame, in a fixed order,
because each check needs the one before it to have passed:

1. **The file's keys and value types.** A key the format does not have, a required key missing, a
   value of the wrong type. The loader reads only the keys it knows, so a misspelt key would
   otherwise be dropped without a word, and a value of the wrong type would escape later as a raw
   `KeyError` or `TypeError` naming neither the step nor the key.
2. **Duplicate step ids.** Without unique ids an edge cannot name its far end.
3. **Unknown tools.** A tool's template says how a rule value is written into each field, and the
   port checks need a tool to ask. The lookup reads only a step's `tool` and `id`, which no rule
   reference can reach.
4. **Rule references** (`${rules.*}`). Settings cannot be read until they are resolved, and
   resolving finishes before anything validates them.
5. **Lanes**, then **containers** (both ways a step can be mis-boxed; see below).
6. **Ports and edges.** A port or an edge that does not exist.
7. **Cycles.** Reachability means nothing in a graph with a loop.
8. **Each step's settings**, against the tool's template and then its own `validate_config`.
9. **Dead ends** (see below).

A tool's `config_template()` takes no settings, and must keep taking none: it is read before the
settings are final, so a template that consulted its own settings would be a loop.

The column references a loaded flow publishes cannot be checked here, because they need frames.
They are checked by the runner when each step's inputs exist.

`load_pipeline` takes a plain mapping because a file format is not part of what a flow is.
`read_pipeline` is the file-reading convenience on top.

## YAML is read with the C loader when there is one

PyYAML is imported inside `read_pipeline`, so importing the library does not import it. Its C safe
loader is used when PyYAML was built with it, which reads a large file about eight times faster;
otherwise the pure-Python safe loader reads the same document.

## A refusal that mentions old names

When a document is refused and it still uses names retired in a release, the refusal is usually
about an old name. So the loader replaces it with a short account: one sentence, the migrate command
(naming the file when it is known), and one line per kind of rename. It then loads the renamed
document too. If that is also refused, the problem is not an old name, and that refusal is shown
after the account.

## A port named twice is refused first

A port holds one frame, so a second use of a port name in a step would quietly replace the first.
The loader's other port checks work on sets, where a repeated name disappears, so they would say
nothing. This check also runs first for a second reason: an unwired duplicate trips the "nothing
feeds this port" refusal, which would send the author to the arrows when the thing to fix is the
step's settings.

The ports a step reads are worked out from its settings as written, with no defaults, because edges
are resolved before settings are validated. The runner asks the same question of the same settings,
so both get the same answer.

## Dead ends

**Decided.** A step whose output nothing reads, and whose tool is not an `output` or an `assert`, is
refused at load. Such a step computes rows for nobody.

Removing a dead end can leave the steps that fed it unread, so the rule is applied again until
nothing more dies. `dead_ends()` answers for the whole chain at once: first the steps nothing reads,
then those that fed only them, round by round, each round in file order. It works on the raw
document, before rules are resolved, so it can answer for a file the loader would refuse. Its
`without=` argument asks "what if this edge were removed", which is how to plan removing an output
before deleting steps one refusal at a time.

## Containers

A container is a line drawn round steps that do one job, with a sentence (`title`) saying what the
job is. Six hundred boxes cannot be read by following them, and a reader looking for "where does the
extract get its header row back" is looking for a job, not a tool. The title is a sentence because
the whole value of the feature is a reader getting one sentence instead of the six boxes it stands
over.

A container changes no frame and reaches no tool, and it is left out of the fingerprint (see
"Fingerprints"). Its steps are listed in file order, the only order a container has; a set would draw
the same container differently on each render.

What enters a container is every port made outside it and read inside it, listed once however many
steps read it: two steps taking the same port are one lot of rows arriving. What leaves it is every
port of a member that a non-member reads, or that nothing reads. The second half matters: rows that
stop at a member have still left the container (to a file, a report, or nowhere), and a container
whose last step is a write would otherwise claim nothing came out of it.

### A container must have a title

A blank title counts as missing. A title of spaces passes every check that asks whether the key is
there and draws a box with nothing written on it, which is the same silent failure as falling back on
the id: the container looks finished and nothing says a sentence was owed. An id is not a title.

### What the loader checks about containers

- **A step names a container nobody declared.** The container is drawn one step short, with a stray
  node beside it.
- **A declared container no step names.** An empty rectangle with a title over nothing.

These two are mirrors, and the pair is the point: neither says anything is wrong, and on six hundred
steps nobody finds either by looking.

- **A container whose steps sit in more than one lane.** A lane is a horizontal band. A box across
  two lanes is wrong whichever way it is drawn: stretched over both, it swallows the steps of the
  other lane that are not in it; drawn round one, it leaves a member outside. It is refused rather
  than redrawn, because the fix is one line (move the step, or split the container) and only the
  author knows which was meant.

**Rejected: requiring a container's steps to be connected.** The commonest container is "read the
three workbooks": three reads that touch nothing of each other and meet in a join outside the box. A
connectedness rule would refuse exactly that.

## A step holds no defaults

`Step.config` holds the settings as written, with rule references resolved and nothing added. The
file a person edits and the object the page draws must stay the same thing; a reader seeing a field
nobody wrote is how a page starts misstating what is there. Defaults are filled by the boundary, at
the one moment they are needed.

`Pipeline.registry` is not part of equality: two flows loaded from the same document against the
same catalog are the same flow, whatever registry object answered for them. A flow built in code
without a registry gets the shipped one.

## Rule references

**Decided.** A `${rules.name}` reference is replaced by the rule's value before the settings are
checked.

- **A reference that is a whole field** takes the rule's own value and type, so a list stays a list.
- **A reference inside other text** is written into that text, spelled for the kind of field it sits
  in. The template says what kind that is.
  - In an **expression** field the value is written in the expression language's own spelling:
    text quoted and escaped, a number as written, `true`, `false`, `null`, a date as
    `date'YYYY-MM-DD'`, a list in parentheses (`('DRAFT', 'SUBMITTED')`), because square brackets in
    an expression mean a column. The person editing a pipeline file is stating a value, not writing
    code, and should not need to know that `O'Brian` is spelled differently from `Parkside`. An
    empty list is refused, because `in ()` matches nothing.
  - **Inside quotes the author wrote**, as in `[code] = '${rules.kept_code}'` or
    `[month] >= date'${rules.start_month}'`, the author asked for text, so the value is written as
    text and escaped. A list there is refused, with a pointer to `in`. Whether a reference is inside
    quotes is read from the string as the author wrote it, never from the text produced so far,
    because one field may hold two references wanting opposite spellings. A quote inside a column
    name in brackets (`[client's name]`) opens no text, and a doubled quote is one quote.
  - **Every other field** (a pattern, a path, a sheet, column or table name) gets the value and
    nothing else. A quote written into one of those is a character to match or part of a file name,
    not an escape.
- **Written as text**, a value must have one text form: text as it is, a whole number in digits, a
  date as `YYYY-MM-DD`. Anything else is refused rather than written the way Python prints it, which
  would turn `no` into `False` and a list into `['a', 'b']`.
- **A field the template does not describe** is written as is and never refused here. An undeclared
  field is a misspelled name, and the template check says so; a refusal about rules would send the
  author to the wrong half of the file.

The value and its template entry are walked down together, because the fields this matters to are
nested: a rule's conditions, an aggregation's value and a new column's name all sit two levels down,
next to an expression wanting the opposite spelling. Where the template stops, everything below is
written as is.

The `{scratch}` token has no dollar sign and is never touched here. It belongs to the conformance
check.

## One rename table, keyed by where a name sits

**Decided.** `renames.py` holds the one table of names retired in a release and what each is called
now. The loader reads it to explain a refusal; the migration reads it to rewrite old files.

**Why keyed by position.** A name is renamed where it sits: which tool, which field, which part of
the file. The same word somewhere else is not a retired name. So the table is keyed by tool and by
path, and `find_renames` walks a parsed document the same way. A path such as `tables[].key` names
the `key` member of each entry of the `tables` list. The parts before its last one are written in
the new names.

Some settings must now be stated where they used to have a default (`STATED_NOW`). Where the file
alone shows which value keeps a step doing what it did, the migration writes that value in.

## The migration edits text

**Decided.** `migrate` edits a pipeline file as text. It never loads the file and writes it back.
Only the characters of each renamed name change, so comments, layout, quoting and `why` text stay as
they were.

**Checked.** After editing, the new text is read back and compared with the old document with the
renames applied to its data. If the two differ, the file is refused and left as it was, rather than
written wrong.

A YAML node written once under an anchor and reused by an alias is one node, reached from every step
that uses it. A node shared by steps that would edit it differently is refused, for a person to
untangle.

**Rejected.** Load and dump. It would lose every comment and reflow the file.

## Fingerprints

**Decided.** `Pipeline.fingerprint(target)` is one string answering "is the configuration that
produced this port still the configuration in the file?". It covers the step and every step behind
it.

- **The closure, not the step alone.** The rows on a port are the product of everything upstream. A
  step whose own settings are untouched produces different rows once a filter feeding it keeps
  different ones. A hash that could not see that would report a match and hand back yesterday's
  numbers under today's rules.
- **Four things per step:** `id`, `tool`, `config` and `inputs`. Left out: `lane`, `stage`, `why`
  and `container`. None of those reaches a tool, so a frame produced under one `why` is the same
  frame under another. A cache invalidated by an improved comment is a cache nobody trusts, and so
  one everybody works around. Boxing steps in a container is a line on the page; if it threw away
  every cached frame behind them, the first person to tidy the page would pay with an hour of
  re-running, and the second person would not tidy it.
- **A port's fingerprint covers the port's name**, so a frame cached on the wrong side of a split
  (its `true` frame under `false`) does not match. A step with one output port has nothing to mix
  up, and the step's fingerprint and its port's are the same string.
- **Steps are hashed in id order**, not file or run order, because where a step sits in the file
  changes nothing it computes. A hash that moved when two independent steps were reordered would go
  stale on a tidy-up.
- **Canonical JSON with sorted keys, hashed with SHA-256**, not Python's `hash()`, which is salted per
  process for strings. The fingerprint is written down in one run and compared in another. Sorting
  the keys makes settings typed in a different order the same settings.

**The limit.** A fingerprint covers configuration, not data. See "A fingerprint covers the
settings, never the data" below.

### Values JSON cannot write

A date or timestamp in a configuration is written as ISO text for the fingerprint. YAML hands back a
date for an unquoted `2026-04-01`. That cannot arrive through `load_pipeline` today: every field the
catalog declares is typed, none is a date, and a bare date in a pipeline or rules file is refused as
the wrong type. The branch is there for a `Pipeline` built in code, and for the day a tool declares a
date field.

Any other value is refused rather than handed to `str`. `str` of an arbitrary object is a repr
carrying a memory address, which differs between the run that wrote a fingerprint and the run that
compares it. Every cache would then report itself stale, and the failure would look like the feature
not working.

## Cycles

The walk order is a topological sort with ties broken by file order. When the sort cannot place
some steps, each of them reads at least one other unplaced step. Walking upstream along those edges
must come back to a step already seen; the steps from there on are the loop, and the refusal names
just those, not everything that depends on it.

## Planning a run before it starts

`walk.py` works out, before any step runs, which steps to run, checks the frames handed in through
`sources`, `cached` and `keep`, and works out when each frame can be freed. Every refusal here comes
before the first step runs.

### `until` refuses an unknown or empty list

What runs is the steps `until` names and every step that feeds them, in the order a whole run would
run them. Both lists on the trace (asked for, and run) are the flow's run order filtered to a set of
ids. So a step that runs sits where it sits in a whole run, a step reached two ways runs once, and the
trace reads the same whatever order or kind of collection the caller wrote.

A step id the flow does not have is a typo. Running nothing and running everything are equally
defensible readings of one, and both leave an author looking at a result that does not answer their
question, so it is refused. An empty collection is refused for the same reason: it almost always
comes from a comprehension over something that turned out empty, and a trace with no events reads
the same as a run that failed on its first step.

### `sources` and `cached` are different doors

Both arguments hand frames to a run, and a typo must read the same in either: a badly spelled key, a
step the flow does not have, and a port that step does not declare are worded once and told apart
only by naming the argument the caller wrote.

What they disagree about is the point of having two:

- A **source** is the caller stating this run's input. It must be an output of a step that reads
  nothing, the only kind of step that can be handed a frame without the frame quietly becoming the
  flow that ran. It is checked against nothing, because nothing is upstream of it.
- A **cached** port is a claim that a computation which already happened may be skipped. It must be
  a computed port, and its fingerprint must match.

**Rejected.** One argument for both. One frame with two doors and two different guards would leave
every later question (was this run resumed? is this trace evidence?) answerable both ways at once.

What a step reads is asked of its tool with the step's own settings, because a tool that takes its
ports from its settings names none in its attribute even when the step reads three.

### What `cached` refuses

Structure first, fingerprints after: a fingerprint cannot be taken of a step that does not exist, and
a refusal about a stale cache would send the author to the wrong problem.

- A source port. The refusal names `sources`.
- A step cached on some but not all of its output ports. It produced them all in one pass, so getting
  the rest would mean running it, its ancestors and the reads they open, which is the whole cost the
  cache was meant to avoid. It would also leave two answers for one step, one from a previous run.
- A fingerprint that no longer matches the file.

### A cache skips the steps behind it

Skipping only the cached step would save one step out of fifty, still open every file the flow
reads, take as long, and report that it had resumed. So the walk is worked out backwards from what
the run is for: the steps `until` named, or, for a whole run, the steps nothing else in the walk
reads. Every input edge is followed back, except through a cached step, which reads nothing this run.
What survives is every step a running step still needs. An ancestor that two other steps read is not
dropped just because one of them was cached.

A cache that saves nothing is left alone rather than refused. It is not a mistake, and a refusal
would send the author hunting for one.

### When a frame is freed

Under `retain="live"`, a port is freed after the last step that reads it. That is read off the wiring
alone. Under a partial run, "last" is judged within the steps that will run: a reader that will never
run cannot be a port's last use, and counting it would leave a source frame, often the largest in the
run, never freed. A port nothing remaining reads is freed at the step that produced it, for the same
reason: a final frame kept "just in case" is usually the largest one in the run. A port in `keep` is
never freed.

## The runner

`run_pipeline` walks a loaded flow and leaves behind one thing a failed run can be debugged from: a
frame for every port and one event for every step, whichever step the run stops on.

### One walk, two readers

**Decided.** Events are sent to a `sink` the moment they exist and are also kept, in order, on the
returned trace.

**Why.** Two readers share the same record. Someone watching a scheduler's log at three in the
morning wants each event as it happens, worded so a failing step explains itself without a stack
trace. A tool that slices and draws a run afterwards wants every frame, keyed by port so that a
split's two sides and a join's three outputs never fight over one key.

**Rejected.** A separate record for each reader. Two accounts of one run would have to be squared
against each other.

### The log sink writes messages as they are

`logging_sink` logs each event's own message at the event's own level. It never rewords: a sink that
reworded would give a reader a second account of the same run. A warning goes to `warning`, so an
empty frame a later step reads is not lost among the info lines.

A sink that raises is logged once on the runner's own logger and never ends the run it is watching.

### Retention: `retain="all"` and `retain="live"`

**Decided.** Two modes. `"all"` keeps every frame. `"live"` frees a port's frame the moment nothing
left in the run will read it. `keep=` names ports to hold on to anyway.

**Why.** The two readers pull in opposite directions on memory. Every intermediate frame is what a
real run must not hold for long, and what makes a run inspectable afterwards.

When a frame is freed is worked out from the wiring alone, so it is a property of the flow, not a
choice a caller could get subtly wrong. Both modes report identical counts, shapes and messages.
Under `"live"` the last port of a flow, and the ports of the steps `until` names, are freed like any
other unless named in `keep`.

### Refusals come before side effects

**Decided.** As many refusals as possible happen before the first step runs: every refusal about the
run's own arguments (`retain`, `until`, `sources`, `cached`, `keep`, `base_dir`) and every leftover
`${...}` placeholder.

**Why.** A refusal that arrives after several steps have already written somewhere is not a refusal.

**The exception.** A column a step's settings name is checked against that step's own input frames
the moment they exist, not for the whole flow up front. No tool declares the columns it produces, so
an up-front pass could only reach the steps fed directly by supplied frames. This check, and a type
error inside an expression, can therefore refuse after earlier steps have written.

The missing-column refusal lists each input port's columns separately, because a step with two
inputs cannot say which side was meant.

### The placeholder check reads every step

The scan for a leftover `${...}` reads every step, not only the ones `until` reaches. A leftover
placeholder means the file was never fully resolved, which is a fact about the file, not about this
run. A scan narrowed by `until` would call the same file clean today and unresolved tomorrow for a
reason nobody touched. The scan looks inside lists and mappings at any depth.

### Sources are handed in, not read

**Decided.** `sources=` hands a frame to a step with no inputs, instead of letting it open its file
or database.

**Why.** Every step with no inputs opens a file or a database. A run driven from tests, a notebook, or
an upstream job that already has the frame in memory should not have to write it out for a reader to
open again.

**Refused.** A frame for a step that has inputs: accepting one would let a caller substitute the very
thing a run exists to prove happened. A source step must be supplied on all its ports or none. A
source for a step `until` never reaches is accepted silently, so a caller can build one set of
sources for a flow and still cut the run short.

### `until` runs the closure, not a prefix

**Decided.** `until` names one or more steps. What runs is those steps and every step that feeds
them. Nothing else runs, and steps outside leave no event, no shape and no frame.

**Why.** A pipeline is built by growing it. A flow that can only run whole cannot be grown: the only
way to look at step ten would be to survive step six hundred.

**Rejected.** Cutting the run order at a position. A step written early sits early in the order
whether or not the named step needs it. With a closure, naming the last step of a flow still leaves
out a side branch nothing on the way reads.

A step that ran and produced nothing and a step that was never asked to run are opposite facts, so
skipped steps leave nothing on the trace. Naming a step together with one of its ancestors is a
union; nothing runs twice. `None` runs everything.

### `cached` starts a run where `until` stopped one

**Decided.** `cached=` takes frames an earlier run left on computed ports, each with the fingerprint
of the steps that produced it. A step whose output ports are all cached does not run, and nor do its
ancestors unless a running step still reads them.

**Why.** That is the whole saving: "run the first ten steps, look at them, write the next ten". The
cost avoided is the files and databases those ten opened.

An empty mapping is accepted, unlike an empty `until`: resuming from nothing is a complete run, which
is what a caller who builds a cache from whatever is on hand should get when there is none. A cache
entry the run never reaches is accepted and leaves no mark.

A cached step reads nothing in this run, so its inputs are not gathered and its columns are not
checked. A missing column on a frame that was never read is not a fact about this run.

### A fingerprint covers the settings, never the data

The fingerprint cannot cover data. The expensive reads are the ones an author wants to skip, so
hashing what they returned would mean doing them. A cache built from yesterday's database read and
resumed against today's is a wrong number nothing can see: every setting matches, nothing is refused,
and the figures are stale.

So `cached` is for building a flow, not for producing figures, and a run that used one is not
evidence about today's figures. `PipelineTrace.cached` lets a report or a scheduled job refuse such a
run without asking the caller.

### What a step's event says

- **Every output port's row count**, so a split reports both its sides.
- **Where the rows came from**, in three different words: `produced`, `supplied, not read`, and
  `read from a cache, not re-run`. A supplied step and a cached step make different claims. Inside,
  this is one three-way value, not two booleans, because "ran" and "skipped for two different
  reasons" is one question with three answers.
- **The tool's own notes first**, then anything the runner noticed. A tool knows things about its
  work that nobody watching its ports could reconstruct, such as how many values it could not read.
- **An empty frame that a later step reads** is a warning, because otherwise a quiet failure and a
  quiet day read the same in a log. This asks the whole flow, even under `until`: an author growing
  a flow a few steps at a time needs to be told. Freeing a frame is a fact about this run, so the
  release schedule asks only the steps that run.
- **A failing step's event** is sent to the sink before the exception leaves `run_pipeline`, so a
  crashed run still has a log of everything up to it.

### A caller can watch every step through `run_tool`

The runner calls the boundary as a bare `run_tool(...)` through its own module name. A caller may
replace `runner.run_tool` to watch every step's output as the run goes. Keep the call in that form.

### `base_dir`

Each tool's path fields (marked `file_path` in its template) are joined onto `base_dir`, so a flow
whose paths are relative to its project folder runs from any working directory. Left out, a relative
path is read from the working directory. An absolute path is never changed.

## The trace

### A cached frame carries its fingerprint

**Decided.** `CachedFrame` holds two things: the frame an earlier run left on a port, and the
fingerprint of the steps that produced it (`Pipeline.fingerprint(port)`). `run_pipeline` compares the
fingerprint with the flow it is about to run, before it runs any step.

**Why.** Neither half is any use alone. A frame without a fingerprint is a number with no history.
The danger of a cache is not one that fails to load. It is one that loads perfectly against a flow
somebody has since edited.

It is frozen because it is evidence of a run that already happened.

### What the trace says about itself

A trace is read by someone who did not produce it and cannot ask the caller anything. So a trace says
on its own face what kind of run it was.

- **`until`** is what a partial run was asked to reach, and is empty for a whole run. Drawing a
  seven-step part of a nine-step flow as the whole flow would tell a reader two steps do not exist.
- **`cached`** names the ports taken from a cache, in the order the run reached them. It is the only
  thing that tells a computed run from a resumed one, because a resumed run's figures look as
  plausible as any other's. A cache offered but never reached does not appear.
- **`shapes`** is complete whatever `retain` says. How large a freed frame was is what a failure
  needs. `frames` holds only what retention kept.
- **The mappings are read-only.** A trace is evidence of what happened.

### Per-row status stays with the step that made it

`status` holds, for a port where a step emptied or changed values, one row of status per row of that
port. Ports with nothing to report have no entry, which on a real flow is almost all of them.

**Decided.** Status does not travel downstream. A step that passes on rows an earlier step emptied
reports nothing, because a status is a fact about what a step did.

**Rejected.** Carrying status forward. One problem would mark every step after it, and the last step
of a flow would look like the culprit.

A status is freed together with the frame it describes; kept alone it would point at nothing.
`shapes`, a summary rather than a reference, stays.

### A failure carries the partial trace

`StepFailedError` holds `step_id`, the failing step's `event`, and the `trace` up to the failure,
frames included, so a caller can still save or show what ran. Its `__cause__` is the tool's own
error, unchanged.

It is a plain exception class, not a frozen dataclass. Python and pytest write to an exception's
`__traceback__` and notes, and a frozen dataclass refuses those writes with a second error. It
defines `__reduce__` so it can be pickled and sent to another process.

`ColumnCheckError` is a kind of `StepFailedError` for a step whose settings name a column none of its
input frames have. It is raised the moment the frames exist, before the tool runs.

### A frame shape keeps column names

`FrameShape` keeps the column names, not only a count. Under `retain="live"` a freed port's frame is
gone, but its columns are still worth drawing, and a width alone cannot be drawn.

## Focus

**Decided.** `slice_trace` narrows a finished run to one entity (one client, one invoice) and returns
a view of every port filtered to that entity's rows. It reads the run; it never re-runs it.

**Why.** A run re-run over one entity's rows answers a different question. A total sums over every
row and a join matches across a whole frame, so the re-run gives numbers that look reasonable and are
wrong. Reading the finished run also means the same run can be sliced again for the next entity and
answer the same way.

### One value, and no per-port column

`FocusSpec.identity` is a list of candidate column names, tried in order at every port; the first one
a port has is used, and never more than one. A detailed frame and a summary often name the same
entity differently, so one name would not do. Nothing checks that two candidates never sit on one
port together.

**Rejected: naming a column per port.** A slice applies one value, the entity's own. A key column
named at a summary level would be filtered for a value it never holds and match nothing, silently.

The only per-port thing a focus may say is a `RelatedKey`: "reach this port, which names no entity,
through a column it shares with another port". The key's values are learned at the other port,
filtered to the entity's rows there. So the value a related port is matched on is always learned,
never the one a reader typed.

### Four answers at a port

Confusing any two gives a believable wrong number, so each has its own status:

- `matched`: the port names the entity. A count of 0 is a fact about the entity.
- `related`: matched through a learned key, not the entity's own value.
- `no_grain`: the port has no column naming any entity. This is a fact about the port. Reporting it
  as zero rows would send a reader hunting a filter that never fired, so the count is `None`.
- `not_retained`: the run freed this frame under `retain="live"`. A fact about the run, count
  `None`. Its size and columns are still reported.

`no_grain` is checked before `not_retained`, because whether a port names an entity is known from
its shape alone. A port's frame is `None` exactly when its count is, so a shared reference table is
never drawn under one entity's name and a freed frame is never made up.

A port reached through a related key offers nobody to the picker (`available_entities`): a row there
belongs to several entities at once, and listing it under one would double-count. An entity's row
count is only kept per port, because the same entity is one row at a summary level and several at a
detail level; one number across both would be wrong at every level but one.

### A value nobody holds is refused

A view showing zero rows at every port looks the same as an entity the flow dropped at its very first
step, which is the wrong answer focus exists to prevent. So a value no matched, kept port holds
anywhere is refused. The refusal fires only when it can be backed up by at least one such port; a
`retain="live"` run keeps none, can refute nothing, and is sliced as it is.

When a focus is refused on a run that failed, the failure is almost always the cause (the ports the
focus needed were never reached), so the refusal opens with the failure.

`MissingEvidenceError` is for a related key whose source port the run never reached. A port that
ran but was freed is shown as `not_retained` instead, because that is a fact worth showing, not a
request that cannot be answered. `RelatedKeyError` is a mis-declared key (absent at one end, or a
source port that names no entity): fix the focus, not the flow.

## Preview

**Decided.** A finished run's ports are written to Parquet once, beside the page, and served a window
of rows at a time. Filtering happens against the whole port and only the window reaches the browser.

**Why.** A page draws a fixed number of rows per port, because a page that draws them all is one no
browser will lay out: the twelve ports of one real flow came to 113 MB of text. That cap puts every
question of the form "does any row anywhere hold this?" out of reach: a search over the drawn rows
answers a question about the page, not the data. Serving windows keeps the memory a reader costs
from growing with the flow.

`PortStore` holds no frame in memory. Each request scans the Parquet lazily, so a port of five
million rows costs the same as one of five hundred, and two readers do not queue behind each other.

Every value crosses as text, with its type named separately. A date, a decimal and a large integer
each lose something different on the way through JSON, and the page shows text anyway. Converting
once, here, beats three roundings nobody chose.

A port that was freed is absent, not empty. Showing it as a port with no rows would look the same as a
step that produced none: the difference between a quiet day and a lost figure.

`preview.py` uses the standard library's `json`, not `orjson`. The library is imported by other
projects, so every module-level import is one they must carry. Measured cost: a 625 KB run manifest
parses in 4.6 ms instead of 3.0, read lazily once per page.

### Searching a saved run

**A filter is a substring of the value's text, never an expression.** The expression language exists
for files a non-developer edits, with checks to match. A search box is typed into casually, so it
gets the construct with no failure modes: text that is in a value or is not. A reader who needs a
real condition has the `filter` tool, which is a checked step in the flow. Values are matched as the
page shows them: 2026 typed against a date column means the text, not a number.

**The status level is its own parameter**, not one more column filter. Column filters are substring
searches, which suit a half-remembered value and not a list of three words: `warn` would match
`warning`, and `error` would match every row of a column that holds the word. So a level is checked
against the published levels and refused if it is not one. A level asked of a port with no status is
refused too, because every row and no rows would both be plausible answers to a wrong question. It
is not reachable as `f.level`, because a port may well have a column called `level`, and one control
would then mean two things on two ports.

**All narrowings apply to the same rows.** A reader types a value in one box and picks a level in
another and expects the rows that meet both. `match` (exact values, used to follow one entity
through a saved run) narrows together with them.

A request for an unknown port, an unknown column or a window that runs backwards is refused by name,
because answering with nothing would look like a port that matched nothing.

### Status is saved beside the port

A port's per-row status is written to its own Parquet file beside the port, not as extra columns
inside it. The port file is what a reader is shown as the data, and must show what the step
produced and nothing it said about it. The two are joined by row position only, so the manifest
records the status's row count, and the store checks it against the port's before reading. Without
that check, a hand-assembled or truncated directory would be padded with nulls and draw a level
nobody published.

A port with nothing to say gets no file and no entry; an empty file per port would cost a read each
to say nothing. `status={}` is a complete run, not a refusal, so a caller assembling statuses from
whatever its run produced does not have to write the condition.

`persist_trace` takes the whole trace so the statuses cannot be forgotten. A directory written
without them loads, serves and draws exactly like a run whose steps changed nothing, and the gap
shows only when a reader asks which rows a warning covers. A status is written only for a port whose
frame was kept: marks beside rows nobody can read point at nothing.

### The status travels with the rows

`Window.status` is `None`, not a tuple of `ok`s, for a port no step said anything about. The page
reads that to decide whether to draw a status column at all. Almost every port in a real flow
changed nothing, and a column of ticks on all of them is one a reader learns to skip, which costs the
flagged ports their attention.

The marks are sliced with the rows, as one struct column concatenated by position, and never worked
out again afterwards. A window that took the first marks for rows at a later offset would be right
on page one and, on every page after, put a warning on a row that never had one. A join on some
column either side happens to have would be a different question with a plausible answer.

The status column's name is lengthened with `_` until no column of the port has it, because a port
may have a `status` column of its own, and a collision would overwrite a column the reader came to
read. The page is handed only the port's own columns, so the marks are never shown as data.

### Panels are separate files

Each step's panel (what a reader sees on opening the step) is its own file. The page once carried all
672 panels of a real flow: 7.76 MB of an 8.98 MB document, laid out before the browser could paint
anything. One file each, not one file holding every panel, because the service reads a run's
directory per request and a single file would be read whole to answer for one step.

A step id arrives in a query string, so it is looked up in the run's index (`panels.json`), never
turned into a path. An unknown step is refused before any file name exists. File names are
percent-encoded so any step id is one file in one directory. Names read from the index are still
checked to stay inside the run's directory, because the index is a file someone could edit. A step
with no panel is refused by name: a blank answer would draw an empty pane over a step that has rows,
and a missing file would be a stack trace where a sentence belongs.

The index files are read when first needed, not when the store is built, so a store built to hand
over one panel does not parse a 625 KB port index. A mistyped directory is still refused at once.

### A local viewer with no login

`serve` starts one server for all of a project's runs, each at `/<folder>/`, because runs are read
side by side and a figure in one is chased into another. Runs are found by looking for folders with
a `ports.json`, not by registration, so a second flow is one more folder.

**It has no authentication and no encryption.** Every row of every port is readable by anything that
can reach the port. That is a chosen trade: the alternative is a dependency chain and a login on a
tool one person uses to look at their own run. Putting it where a stranger can reach it means
choosing a different tool. It listens on `127.0.0.1` unless told otherwise; `0.0.0.0` is for a
container whose port is published to its host.

The source route serves only YAML, Python and Markdown files under the source root, never a hidden
file or folder (no `.env`, nothing in `.git`), never an absolute path or one that leads outside.
