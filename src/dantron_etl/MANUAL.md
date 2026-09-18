# Manual: building a data pipeline with this library

Written for an LLM agent. If you are a person, it works for you too, but the ordering and the
bluntness are aimed at an agent deciding what to do next.

This library gives you a fixed set of **tools**. You build a data pipeline by writing a YAML file
that names tools and configures them. You do not write transformation code, and you do not add a
tool because the one you want is missing — read [When a tool really is missing](#when-a-tool-really-is-missing)
before you conclude that it is.

---

## 1. What you are allowed to do

| | |
|---|---|
| **Write** | a pipeline file: a list of steps, each naming one registered tool and its configuration |
| **Write** | expressions, in the small bracket language described below |
| **Read** | the generated catalog, to find out what a tool does and what it accepts |
| **Do not write** | Python that reshapes data outside a pipeline run |
| **Do not add** | a tool, without first proving no combination of existing tools does the job |

The reason for the third row is not style. A pipeline file is the single artefact that defines what
a run does; the moment a script beside it also reshapes the data, the file stops being the answer to
"what happened to this number" and nobody can tell which of the two produced a figure. Scripts may
run a pipeline, compare its output against something, and render it. They may never reshape it.

---

## 2. The rule that matters most: a new tool is a last resort

Three capabilities were declared missing during this library's own first real pipeline. Every one of
them turned out to be **two existing tools in sequence**:

| what looked missing | what it actually was |
|---|---|
| a date formatted as text | `cast_columns` to `String`, then `formula` with `regex_extract` |
| a grid with zeros in the empty cells | build the rows first with a `join`, then `formula` with `coalesce` |
| a number inside a sentence | `cast_columns` to `String`, then `formula` with `concat` |

The failure mode was always the same: check whether **one** tool does the job, find that none does,
and stop. Before concluding a tool is missing, answer these in order:

1. Can one tool do it? Search the catalog by what you want to *end up with*, not by the verb you
   first reached for.
2. Can **two** tools in sequence do it? Most "missing" capabilities are a type change followed by a
   computation, or a reshape followed by a fill.
3. Is it a *column* computation rather than a reshape? Then it is an **expression function**, not a
   tool — see [the tool-or-function rule](#43-tool-or-function).
4. Only then: is it genuinely a new reshape of the frame?

---

## 3. Finding out what a tool does

Three sources, in order of how much you should trust them:

**The generated catalog** — a JSON file produced from the live registry, holding every tool's
description, its full configuration schema, and worked examples with real input and output frames.
It cannot drift, because it is generated from the code and every example in it is **executed and
compared** by the conformance suite. This is your reference. Ask whoever set up your environment
where it is written; it is produced by a script in the repository, not shipped in the package.

**The registry, at runtime:**

```python
import dantron_etl
from dantron_etl.tool import REGISTRY

for tool in REGISTRY.tools():
    print(tool.name, tool.category, tool.input_ports, "->", tool.output_ports)

tool = REGISTRY.get("group_by")
tool.config_template()   # every field: type, description, required, default, enum
tool.help()              # detailed description, worked examples, notes, per-field help
```

`tool.help().notes` carries the rulings that are not obvious from the schema — what happens to
nulls, what is refused and why. Read them. They are where the money bugs live.

**This manual** — the rules that span tools, which no single tool's page can tell you.

### 3.1 The tools that ship

29 tools, by category:

| category | tools |
|---|---|
| `input` | `read_csv`, `read_excel`, `read_table` |
| `preparation` | `cast_columns`, `filter`, `limit_rows`, `record_id`, `remove_duplicates`, `rename_columns`, `select_columns`, `sort` |
| `transform` | `assemble_date`, `case_when`, `cross_tab`, `date_diff`, `expand_date_range`, `formula`, `group_by`, `unpivot` |
| `combine` | `join`, `union` |
| `text` | `normalize_text`, `regex_extract_groups` |
| `assert` | `assert_rows` |
| `output` | `publish_tables`, `write_csv`, `write_excel`, `write_table`, `write_workbook` |

"Preparation" selects among what is already there. "Transform" produces something that was not.

---

## 4. How a pipeline file is shaped

```yaml
name: Monthly figures

lanes:
  - {id: source,  label: "As loaded",   order: 0}
  - {id: figures, label: "The figures", order: 1}

steps:
  - id: orders
    tool: read_table
    lane: source
    stage: read
    why: >
      Every order as the system holds it. The `why` is not a comment -- it is rendered beside the
      step and is what a reader checks a number against.
    config:
      connection: DATABASE_URL      # the NAME of an environment variable, never a value
      schema: sales
      table: orders

  - id: recent_orders
    tool: filter
    lane: figures
    stage: figures
    why: Only the orders inside the reporting window.
    config:
      predicate: "[placed_on] >= date'2025-01-01'"
    inputs: {rows: orders.rows}
```

Every step has `id`, `tool`, `config`, and `inputs` (except a source, which reads no port). `lane`,
`stage` and `why` drive the rendered walkthrough.

**`why` is required to be useful.** It is the sentence a reader sees next to the number. Write what
the step decides and why that is the right decision, not what the configuration already says.

### 4.1 Ports

A step's inputs and outputs are **named frames**, not a single value. `inputs` wires one upstream
port to one of this tool's input ports:

```yaml
inputs:
  left: orders.rows
  right: customers.rows
```

Every declared output port always exists, empty-but-typed when nothing reaches it. Wire the ports
you need and leave the rest unwired — nothing is lost silently, because an unread port is visible in
the run.

The ports that surprise people:

| tool | in | out |
|---|---|---|
| `filter` | `rows` | `true`, `false` |
| `join` | `left`, `right` | `matched`, `left_only`, `right_only` |
| `remove_duplicates` | `rows` | `unique`, `duplicate` |
| `expand_date_range` | `rows` | `rows`, `unexpanded` |
| `limit_rows` | `rows` | `rows`, `dropped` |
| `union` | `top`, `bottom` | `rows` |

`join` with `how: left` still writes to `matched` — `how` decides only which rows reach `matched`,
and the orphan ports answer their own question either way. So a left join reads `matched`; the rows
that paired with nothing are *also* on `left_only`, which is where an orphan check comes from free.

`expand_date_range`'s second port is not tidiness. A row it cannot expand does not disappear — the
obvious implementation turns it into **one row with a null period**, which looks like a real period,
carries the whole figure, and lands in a report as a month that does not exist. `unexpanded` is
where those go.

### 4.2 The expression language

Columns are `[name]`. The language is parsed, whitelisted and compiled — it is never `eval`'d, and
it never will be, because a pipeline file is something a non-developer edits and commits.

```
[amount] > 0 and [status] = 'OPEN'
coalesce([discount], 0.0)
concat('order ', [number])
regex_replace(lower([name]), '[^a-z0-9]+', ' ')
add_months([starts_on], [months] - 1)
format_number([total], 2, ',', '.')
[placed_on] >= date'2025-01-01'
```

Available functions: `coalesce`, `round`, `year`, `add_months`, `abs`, `lower`, `upper`, `strip`,
`len`, `starts_with`, `ends_with`, `contains`, `regex_extract`, `regex_replace`, `regex_match`,
`concat`, `format_number`.

`format_number(value, decimals, thousands_separator, decimal_separator)` writes a number as text:
`format_number([total], 2, ',', '.')` gives `651,623.80` and `format_number([total], 3, '.', ',')`
gives `123.456,789`. All four arguments are required, because nothing in this language has a
default, and you state both separators rather than having one chosen for you. The last three are
quoted decisions written in the file, never columns. See section 5 for how its rounding differs
from `round`, and note that it is the one function here that works a row at a time — put it at the
end of a flow, on the rows somebody reads, not in the middle of one.

Operators: `and`, `or`, `not`, `+`, `-`, `*`, `/`, `=`, `!=`, `<`, `<=`, `>`, `>=`, `in`,
`is null`, `is not null`.

**Extend it by whitelisting a function, never by growing syntax.** There are no user-defined
functions, no attribute access, no subscripting, no lambdas. Those are refused by name, and that
refusal is a security boundary rather than a preference.

Where the expression is checked against the real frame's schema, a type error is refused before a
row moves — `upper([quantity])` on a number is caught at load, not inside the query engine at run
time.

**Only some fields are expressions, and it matters.** A tool's template says which: `filter`'s
`predicate`, `formula`'s `value`, `case_when`'s `when`, `then` and `otherwise`, `group_by`'s
aggregation `value`, `assert_rows`' `predicate`, and the equivalents in the rest. `help()` prints
the type of every field. Everything else — a path, a pattern, a sheet name, a column name, a schema,
a table — is a plain string and is read by whatever consumes it, not by this language. That
distinction is what decides how a `${rules.*}` reference is written into a field; see section 5.

### 4.3 Tool or function?

The dividing line, and it answers "should this be a new tool?" generally:

> **A tool reshapes the frame. An expression function computes one column.**

| capability | home |
|---|---|
| extract one group from a pattern | `regex_extract`, a function |
| extract *named* groups | `regex_extract_groups`, a **tool** — N columns from one pattern is a reshape |
| replace by pattern | `regex_replace`, a function |
| test a pattern | `regex_match`, a function |

This rule makes the library **smaller**, not bigger, and keeps every configuration unconditional —
no field that only applies when another field holds a particular value.

---

### 4.4 Containers

A flow of any size is unreadable as a list of steps. `groups` names boxes and each step says which
box it is in:

```yaml
groups:
  - {id: read_accounts, title: "Fetch the account list and put its header row back"}

steps:
  - id: accounts_raw
    tool: read_table
    lane: sources
    stage: parties
    group: read_accounts
```

On the drawn page a container is a titled box its steps sit inside, which a reader can close to a
single bar. That is the whole point: a box a reader can shut is a part of the flow they have decided
they do not need to look at today.

**What the loader refuses**, before a single step runs:

| | |
|---|---|
| a `group` naming no declared container | a typo is refused, never quietly ignored |
| a container with no members | it would draw as an empty box |
| a blank or whitespace title | the title is the only thing a closed container shows |
| a container whose steps sit in two different lanes | a box spans one lane; two is a drawing that cannot be laid out |

**How to draw the boundaries.** A container is *one idea a person can name*. If the title needs an
"and then", it is two containers. Three to eight steps is the usual size; a run of near-identical
steps is one container however many it is. Steps in a container need not feed one another -- "read
the three source tables" is a fine container. And not every step belongs in one: a single assertion,
or the one big join everything converges on, reads better loose.

**The title is the point.** It is what a reader sees *instead of* six boxes, so it follows §4b: plain
words, what the steps produce, no vocabulary from this library. Sentence case, no full stop, under
about seventy characters.

A container is a way of reading the flow. It changes nothing about what runs, in what order, or what
any step produces -- `group` is not part of a step's fingerprint (§7b), so boxing an existing flow
does not invalidate a cache.

---

## 4b. How to write a `why`

A finance person reads these on a page while tracing what happened to a number. They are not
comments for whoever edits the file.

The first version of one real pipeline had 672 of them and the reader's verdict was:

> I can see that this is so obviously AI-generated that it's really hard for the average human to
> follow.

Here is what he was reading. It is accurate, and it is bad:

> One row per client and column. The melt is told what it is melting, because the columns to the
> right of the months genuinely are not figures and cannot be named here to keep them out. Those
> come back empty and are dropped by the join below, which keeps only the columns whose header
> stated a date -- so an empty cell here never reaches a published figure, and the count of them is
> reported rather than swallowed.

And the same step, rewritten:

> Turns the workbook's wide layout into one row per client per column, so each monthly figure sits
> on its own row.
>
> The melt is told which columns hold figures. The columns to the right of the months hold notes and
> client names, so they come back empty here and are dropped by the join below, which keeps only
> columns whose header is a date. The count of emptied cells is reported rather than hidden.

### The rules

- **Say what the step does first.** Then say *why* only where a reader would otherwise ask "why is
  this here?" -- a rule somebody chose, a trap being avoided, a figure that would be wrong
  otherwise.
- **One plain sentence is a complete description** for a step that sorts, renames, or narrows to
  the columns the next step needs. Most steps are these. Do not write a paragraph about a rename.
- **Short sentences, one idea each.** Two or three of them is a good length.
- **Everyday words** wherever they mean the same thing.
- **Be concrete.** Name the actual column, file or figure rather than speaking abstractly.
- **Keep the reader's own vocabulary.** The words their business already uses -- order, payment,
  period, exchange rate, account code -- are theirs, not jargon.
- **Cut the library's vocabulary.** A reader does not know or care about the envelope, the port, the
  seam, what is anchored or pinned, what something is measured against, or what "reaches", "lands",
  "carries" or "is swallowed". These read as technical decoration and are the main thing that makes
  a description hard going.
- **No ceremony.** Not "it is important to note", "notably", "crucially".
- **No inflated words.** Not leverage, facilitate, utilise, robust, holistic, paradigm.
- **Do not compress several complicated ideas into one sentence** to be brief. If it needs two
  sentences, write two.

### What not to do

Do not restate the configuration. The configuration is already shown beside the description on the
page, so `"Filters where [type] = 'SALE'"` tells the reader nothing they cannot see. Say what that
filter is *for* -- "Only what the business sold. Money it spent is counted elsewhere."

Do not describe how the step is assembled. A reader tracing a figure wants to know what happened to
it, not which tool was configured how.

Do not claim a check always passes, or that a figure is always present. Descriptions outlive the
data they were written against.

## 5. Rules that bite

Every one of these was measured, most of them the hard way.

### YAML will rewrite your words

- **A plain scalar cannot contain `": "`.** `why: The month becomes text, for one reason: a heading
  is text.` is a parse error, and it is reported thousands of lines from where you are looking. Use
  a block scalar (`>`) whenever the text holds a colon followed by a space.
- **`true`, `false`, `yes`, `no`, `on`, `off`, `null`, `~` and their case variants are not
  strings.** `on: [id]` becomes the boolean `True` as a key. Enum values in this library are chosen
  to avoid all of them, and the conformance suite refuses any new one that does not — but your own
  values are your problem.
- **Backslashes depend on how you quoted the string.** In a double-quoted scalar, `"\\b"` becomes
  `\b`, which is what a regex wants. In a block scalar (`>` or `|`) escapes are not processed, so
  `\\b` stays two characters and **matches nothing while reporting success**. Put patterns in
  double-quoted scalars.
- **A comma inside a flow mapping ends the value.** `{from: total, to: Derived, all contacts}` is
  not a two-entry mapping with a comma in the second value — it is a three-entry mapping, and the
  third key is `all contacts`. Quote any value holding a comma: `to: "Derived, all contacts"`.

### Text

- **Double a single quote to put one inside a string.** Single quotes are the only way to write
  text — double quotes are refused, so there is one spelling rather than two — and a quote inside
  the text is written twice, the way SQL and Excel write it. `'O''Brian'` is the seven characters
  `O'Brian`, and `''''` is one quote. There is no backslash escape, deliberately: a backslash means
  different things depending on how the surrounding YAML was quoted, and that is not something a
  person editing a flow should have to hold in their head. A doubled quote is an escape only when
  the string carries on past it, so `''` on its own is still the empty string.
- **A `${rules.*}` reference is quoted and escaped for you in an expression field, and only
  there.** In a field the tool declares as an expression (section 4.2 lists them), both
  `[code] = ${rules.kept_code}` and `[code] = '${rules.kept_code}'` resolve to a predicate matching
  exactly the rule's value, apostrophes and all, and `date'${rules.start_month}'` puts the date
  inside the tag. Never escape a rule value yourself there: state the value as it really is, and
  the loader spells it. A number, and a list of numbers, stay numbers rather than being quoted into
  text.
- **In every other field the reference gets the rule's value and nothing added.** A pattern, a
  path, a sheet name, a column name, a schema and a table are not expressions, so
  `pattern: "(?P<period>${rules.period})"` gets the pattern itself and
  `path: "out/${rules.holder}_figures.csv"` gets the file name you meant. A quote spliced into one
  of those would not be an escape; it would be a character to match, or a character in a file name.
- **A rule whose value is a truth value or a null cannot sit inside quotes.** Outside them, in an
  expression, `${rules.enabled}` becomes `true`, `false` or `null` — this language's own spelling.
  Inside them you asked for text, and `wanted: no` in a rules file is not text: YAML turned it into
  the boolean before the loader saw it. That is refused by name at load rather than resolved to the
  five characters `False`, which would match nothing while reporting success. Quote the value in
  the rules file.
- **A reference that is the whole field takes the rule's value and its type, in any field.** A list
  stays a list, a number stays a number, and a string arrives exactly as written — including a
  string that is itself an expression, which is how a whole predicate can be stated as a rule.
- **`regex_replace`'s replacement is literal text, not a template.** `$1` is not the first capture
  group; it is the characters `$1`. That is deliberate: the engine underneath reads `$name` as a
  group and deletes it when no such group exists, so a replacement of `$5.00` would land as `.00` —
  which in a library for amounts is unacceptable. The consequence is that a pattern cannot
  rearrange what it matched. `regex_extract` does return capture groups, so pull the pieces out and
  put them back together with `concat`.
- **`format_number` and `round` disagree in the last place, and both are right.**
  `round([x], 2)` on `-15.215` gives `-15.22`; `format_number([x], 2, ',', '.')` gives `-15.21`.
  Neither is a bug. `round` rounds the decimal you wrote. `format_number` writes out the value the
  column actually holds, and a column cannot hold `-15.215`: the nearest double is
  `-15.21499999999999985789...`, which is below the half. The same rule makes `999999.995` read
  `999,999.99`, which looks like a defect and is not — that double is `999999.99499999999534...`.
  If a figure has to be rounded *and* formatted to the same answer, `round` it first and format the
  result. Decide which you want; do not discover the difference in a finished report.
- **`format_number` writes a figure that is not a number as blank, not as a number.** `/` is true
  division, so `[total] / [count]` is infinity on the row where the count is zero and `0.0 / 0.0`
  is not-a-number. Formatting either gives null — an empty cell — because a figure that cannot be
  stated is stated as absent rather than as a plausible-looking wrong one, which is the same answer
  a null already gets. So a blank in a formatted ratio column means the division had nothing to
  divide by. Use `coalesce(format_number([ratio], 2, ',', '.'), 'n/a')` if the report needs a word
  there instead, and guard the division itself with `case_when` if the blank is hiding a fault
  upstream rather than a genuinely empty row.

### Columns

- **A blank name for an output column is refused by every tool that takes one.** Empty, or nothing
  but whitespace -- a non-breaking space pasted from a spreadsheet counts, and is the one you
  cannot see -- in `formula`'s `columns[].name`, in the `output` of `assemble_date`,
  `case_when`, `date_diff`, `normalize_text` and `record_id`, or in `regex_extract_groups`'
  `position_column`. A column called `   ` cannot be referenced later — `[   ]` is not something
  you can write — and it reads as an unnamed column in every rendering, so the step that wrote it
  reports success and the step after it cannot see what it produced. The rule is about a name that
  is *nothing*, not one that is unusual: `" total "` keeps its spaces and is written as spelled.
  The refusal is the same sentence whatever `replace` says, because a blank name is wrong either
  way.
- **`replace: true` on a column that is not there is a refusal, not a no-op.** Both `case_when`
  and `formula` take the flag, and it is a statement that the column is there to be written over. A
  misspelled name fails the run instead of quietly writing a new column and leaving the one you
  meant to repair with its old values. Without the flag, a name the input already carries is
  refused; with it, the column is written where it already was, and the rules read the values that
  were there before the step ran — so a rule may read the column it writes. Reading the old values
  is not the same as keeping them — the column is written in full, so a row no rung reached takes
  `otherwise`, or null when there is none. A ladder that repairs three rows and has no `otherwise`
  nulls every other row of the column.
- **A tool must not mutate its input.** The boundary checks each input frame's schema before and
  after a step and refuses a change, because the walkthrough shows every intermediate and an
  in-place edit makes an earlier step display columns it never produced.

### Types

- **The integer literal in an expression is 32-bit.** `0` and `1` in a `formula` give a narrower
  column than a database's `BIGINT`. It is the right width for a rank or a small count; use
  `cast_columns` when the width itself matters.
- **A spreadsheet cell cannot record a time zone**, so `write_excel` and `write_workbook` refuse a
  zone-aware timestamp. `cast_columns` it to `Date` first, deliberately, rather than letting a
  writer drop the zone quietly.
- **A text file states no types at all.** `read_csv` infers, reading the *whole* column before it
  casts. Declare `column_types` as `String` for any column that is an identifier or a name: a
  customer literally called `3600` inferred as a number stops matching the same name everywhere
  else, and nothing about that failure looks like a failure.

### How a figure is shown

- **Nothing is formatted unless the tab says so.** `write_workbook` writes the figures and leaves
  them under the workbook's own `General`, so `661617.4665392789` opens reading exactly that. State
  `formats` on the sheet entry to get `661,617.47`. The only format written without being asked for
  is on a date column, and that one is not decoration: without it the cell reads as a count of days
  since 1899.
- **A rule names a type or one column, never both and never neither.** A `type` rule is the one
  that reaches a column named by the data — a cross-tab gains a column every period, so
  `{type: Float64, format: "#,##0.00"}` formats next month's column too, and naming columns one at
  a time means editing the pipeline file every month. A `column` rule names one column and beats
  the rule about its type, whichever order the two were written in, which is how one column reads
  to three decimals while every other figure on the tab reads as money. Two rules about the same
  type, or the same column, are refused rather than resolved.
- **Nothing is guessed from a column's name.** A writer that treated any decimal column as money
  is wrong the first time somebody puts a reference number in one, and a writer reading the heading
  for a `%` is wrong the first time somebody renames the column — silently, in a delivered pack.
  So a column no rule matched is left alone.
- **The format code is passed to the spreadsheet as written.** It is not parsed and not checked.
  A code that is wrong is visible the moment the file is opened, and it has not touched the figure
  underneath: `formats` changes how a number is shown and never what it is.

### Ordering

- **Sorting reorders ties unless the tool holds them.** This library's `sort` keeps the incoming
  order within a tie, so the same input always gives the same page. Do not assume any other step
  preserves order — put a `sort` in front of whatever needs one.
- **`first` means "the first row as it arrives".** It is only meaningful after a `sort`. A `first`
  with no ordering in front of it is a coin toss that will land the same way in testing and
  differently in production.

### Paths

- **A relative `path:` resolves against the working directory of the process running the pipeline**,
  not against the pipeline file. Run from a consistent place, or use absolute paths.
- **Config names an environment variable, never a secret.** `connection: DATABASE_URL` is the name
  of the variable; the value is read from the environment. No credential may enter a pipeline file,
  so that a definition is safe to commit and safe to hand to someone else as a template.

---

## 6. Nulls: one rule, several instances

Three tools answer "are two nulls equal?" differently, and each is right:

> **Each tool follows the SQL operation it is named after.**

- `join` mirrors `=`, under which a null pairs with nothing.
- `remove_duplicates` mirrors `DISTINCT`, under which two nulls are one group.
- `group_by` mirrors `GROUP BY`, same.

There is deliberately **no shared setting** for this. One would let a flow configure `join` into
pairing null keys, which fans every null-keyed row against every other on exactly the dirtiest data.

Two more rulings you will meet:

- **`sum` of a group whose values are all missing is null, not `0`.** Zero is a measurement; null is
  the absence of one, and nothing downstream can tell a fabricated zero from a real one.
- **`join_text` of the same group is the empty string** — deliberately the opposite. Joining no
  strings together produces the empty string in any language that has both; adding no numbers
  together produces a total nobody measured.
- **`count` counts rows. `count_values` counts the values that are there.** Other libraries spell
  the non-null count `count`, so a step ported from one silently returns the other number.

---

## 7. Knowing your pipeline is right

Building it is the easy half. A pipeline that runs and produces plausible figures is worth nothing
until something independent agrees with it.

- **Compare against an oracle** — the existing implementation, a signed-off spreadsheet, a hand
  calculation. Compare the **whole frame**, not a total: a total agreeing while the rows behind it
  are wrong is the single most common way this goes wrong.
- **Check the schema first, then the values.** Two frames can hold equal values with different
  types, and a frame comparison that only looks at values will call them identical.
- **Money to the cent, everything else exactly.** Floating point makes `(a/b)/c` and `a/(b*c)`
  differ in the last bit, and a test that fails on that is a test about nothing.
- **Put `assert_rows` in the flow** for the invariants that must hold. It stops the run where the
  step that broke it can still be named, rather than three stages later.
- An assertion over an empty frame passes vacuously — every row of nothing satisfies everything. If
  a branch is legitimately sometimes empty, say so in writing with `allow_empty`.

---

## 7b. Running part of a flow

A large flow is slow to run whole, and while you are working on one corner of it you are running the
other 95% for nothing. Two arguments to `run_pipeline` let you stop partway and start again from
where you stopped.

**`until`** runs a step and everything behind it, and nothing else:

```python
trace = run_pipeline(spec, until="clients_joined")
```

What runs is the *ancestor closure* -- every step the named one depends on, however far back, and no
step that merely happens to be written earlier in the file. Naming a step the pipeline does not have
is refused before anything runs.

**`cached`** hands a port back to a later run so the steps behind it do not run at all:

```python
from dantron_etl.runner import CachedFrame

first = run_pipeline(spec, until="clients_joined")
port = PortRef("clients_joined", "rows")

later = run_pipeline(
    spec,
    cached={port: CachedFrame(first.frame(port), spec.fingerprint("clients_joined"))},
)
```

`CachedFrame` carries the frame **and** a fingerprint, and neither is any use alone. The fingerprint
is `spec.fingerprint(step_id)`, a hash over that step and every step behind it -- their ids, tools,
configurations and wiring. Before the run walks a cached port it recomputes that hash and compares:
if anything upstream has been edited since, the cache is **refused by name** rather than used.

That refusal is the whole feature. A cache that fails to load is a nuisance; a cache that loads
perfectly against a pipeline somebody has since edited is a wrong number nobody can see.

Deliberately **not** part of the fingerprint: `lane`, `stage`, `why` and `group`. None of them
changes a figure, so re-drawing or re-describing a flow does not throw away a cache you would still
have been right to use.

**The loop this is for:** cache the port just before the part you are editing, then run only from
there while you iterate. Run whole once at the end -- a cached run proves what your edit did, never
that the flow as a whole still agrees with its oracle.

### The one thing a fingerprint cannot tell you

It is taken over **the file**, never over the data. `id`, `tool`, `config` and `inputs` -- nothing
else. So a cache survives a change to what the sources hold, and survives it silently:

> Cache a port. Reload the tables underneath. Resume from the cache. The steps behind that port do
> not run, so the new rows are never read, and the run reports a clean resume over yesterday's
> figures.

Nothing catches this, and nothing can: the reads did not happen, so there is no new frame for
anything to compare against. **A cache is only valid for the data that was there when it was
taken.** Throw one away whenever the sources are reloaded, and take a fresh one.

This is not an oversight in the hash. Reading the sources to find out whether the cache is stale is
running the steps the cache exists to skip, so a fingerprint that covered the data would cost exactly
what it saves. The honest answer is a rule a person follows, which is why it is written here.

---

## 8. When a tool really is missing

You have worked through §2 and no combination of existing tools does the job. Then:

### 8.1 Open an issue first

Describe **the shape of the reshape**, not the tool you imagined:

- what is in the frame, and what needs to come out of it
- the sequence of existing tools you tried, and precisely where it stops working
- whether this is one column out of one column (a function) or a change to the frame's shape (a tool)
- a small worked example: input rows, output rows

An issue lets the capability be placed correctly. Several requests that arrived as "we need tool X"
were better answered as a new whitelisted function, or as a field on a tool that already existed,
and one of them was already possible.

### 8.2 Then open a pull request

If you build it, it has to meet the contract every other tool meets. A tool declares:

- `name`, one-line `description`, `category` (one of: input, preparation, transform, combine, text,
  assert, output)
- `input_ports` and `output_ports`, named
- `config_template()` — every field typed (`string`, `integer`, `number`, `boolean`, `array`,
  `object`, `expression`), described, and marked required or given a default
- `help()` — a detailed description, **at least one worked example**, notes carrying the rulings,
  and per-field help
- `validate_config(config)` — cross-field rules only; the boundary already checks the template
- `validate_inputs(inputs, config)` — everything answerable from the **schema**, so a frame of a
  million rows and a frame of none are refused identically
- `execute(inputs, config) -> Ports`

And the standards a reviewer will hold it to:

1. **Every worked example is executed and compared** by the conformance suite. An example whose
   stated output is wrong fails the build. This is what makes the catalog trustworthy, so there is
   no exemption from it — if your tool needs a database, write the example against an in-memory one.
2. **Refuse on the schema, not on the data.** A configuration a frame's *shape* cannot satisfy must
   be refused whether the frame has rows or not, and the message must name the column, what it
   holds, and what was asked of it.
3. **Vectorised only.** No row-by-row iteration inside a tool — it cannot stream, and it is
   thousands of times slower.
4. **Do not mutate the input.**
5. **No client vocabulary anywhere** — not in code, not in docstrings, not in examples. This library
   ships to people who have never heard of your data. A test enforces it.
6. **Name a ruling, do not imply one.** If your tool decides something about nulls, ordering, or
   empties, say which SQL operation it mirrors and put the answer in `notes`.
7. **A guarantee stated in prose is not a guarantee.** This project's most repeated lesson: every
   rule that lived only in a docstring was found violated somewhere when it was finally measured. If
   your tool's documentation claims something, there must be a test that fails when it stops being
   true.
8. **No `eval`, no `exec`, no `getattr` on user text.** Configuration is a file that a non-developer
   edits.

### 8.3 Where

Open the issue or pull request against the repository this package is distributed from — see the
`Source` URL in the distribution's metadata (`project.urls` in its `pyproject.toml`). It is not
hardcoded here on purpose: this package is designed to be lifted into its own repository with no
renames, and a manual naming the repository it currently happens to live in would be wrong the day
after that move.

---

## 9. A worked shape, end to end

A common pattern: reduce a group to one row, keep the detail that would otherwise be lost, and check
the result.

```yaml
  - id: ordered_rows
    tool: sort
    lane: figures
    stage: summary
    why: >
      The order the notes are read in is stated here rather than inherited from whatever order the
      source happened to return. Everything below that says "first" depends on this step.
    config:
      by:
        - {column: customer}
        - {column: reference}
    inputs: {rows: recent_orders.true}

  - id: per_customer
    tool: group_by
    lane: figures
    stage: summary
    why: >
      One row per customer. The references are joined rather than one of them picked: a customer
      with three orders behind a figure needs all three named, or a reader goes looking for money
      the one named order does not hold.
    config:
      by: [customer]
      aggregations:
        - {name: total, function: sum, value: "[amount]"}
        - {name: orders, function: count}
        - {name: references, function: join_text, value: "[reference]", separator: ", "}
    inputs: {rows: ordered_rows.rows}

  - id: every_customer_has_a_total
    tool: assert_rows
    lane: figures
    stage: summary
    why: >
      A customer reaching this point with no total means a row upstream carried no amount at all,
      which is a source problem and not a rounding one. Stopped here, where the step that produced
      it can still be named.
    config:
      label: every customer states a total
      predicate: "[total] is not null"
    inputs: {rows: per_customer.rows}
```
