# Manual: building a data pipeline with this library

Written for an LLM agent deciding what to do next. People will find the same material, written
for them, in the repository's `docs/` folder: a first-pipeline walk-through, concepts, a reference
for the pipeline file and the expression language, a page per tool, and a cookbook. This manual
ships inside the package, so an agent has it wherever the library is installed.

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
compared** by the conformance suite. This is your reference. `dantron-etl catalog` writes it
(`tool_catalog.json`, and a page beside it), and `dantron-etl help <tool>` prints one tool's entry
as plain text, with its first worked example as YAML you can paste. `dantron-etl tools` lists them
all and `dantron-etl functions` lists the expression functions.

**The registry, at runtime:**

```python
from dantron_etl import REGISTRY

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

30 tools, by category:

| category | tools |
|---|---|
| `input` | `read_csv`, `read_excel`, `read_table` |
| `preparation` | `cast_columns`, `filter`, `limit_rows`, `match_one_to_one`, `record_id`, `remove_duplicates`, `rename_columns`, `select_columns`, `sort` |
| `transform` | `assemble_date`, `case_when`, `cross_tab`, `date_diff`, `expand_date_range`, `formula`, `group_by`, `make_groups`, `unpivot` |
| `combine` | `join`, `union` |
| `text` | `normalize_text`, `regex_extract_groups` |
| `assert` | `assert_rows` |
| `output` | `publish_tables`, `write_csv`, `write_excel`, `write_table` |

"Preparation" selects among what is already there. "Transform" produces something that was not.

Four capabilities that each replace a shape you would otherwise build badly:

- **Records that belong together through a chain** (A links to B, B to C, so A and C are one
  group) — `make_groups`. A fixed run of joins follows a chain only as deep as you wrote
  it, and fails silently on a deeper one. Clear blank keys to null first: an empty string is a
  record, and every row sharing one would join a single giant group.
- **Best available match, each side used once** — `match_one_to_one`. A join plus filters cannot
  express it, because whether a candidate is still free depends on which were accepted before it.
  Remove exact duplicate candidates first: under `ties: refuse`, two identical rows tie with each
  other and both keys are refused.
- **Every match in a text, one row each** — `regex_extract_groups` with `all_matches`. Not a fixed
  number of single-match steps, which caps the count.
- **Every tab of a workbook whose name follows a pattern** — `read_excel` with `sheets_matching`.
  Not one read step per tab, which has to be edited every time the workbook gains a tab.

---

## 4. How a pipeline file is shaped

```yaml
name: Monthly figures

lanes:
  - {id: source,  title: "As loaded",   order: 0}
  - {id: figures, title: "The figures", order: 1}

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

  - id: save
    tool: write_csv
    lane: figures
    stage: output
    why: The orders in the window, as a file for the team.
    config:
      path: output/recent_orders.csv
      overwrite: true
    inputs: {rows: recent_orders.true}
```

A file needs `name` and `steps`. Every step has `id` and `tool`, and usually `config` and `inputs`
(a source reads no port). `lane`, `stage` and `why` drive the drawn page. Every step's output must
be read by something, or the file is refused: a step nothing reads is a dead end. A writer
(`write_csv` and the other `output` tools) and `assert_rows` count as reading.

The loader refuses any key it does not know, at every level: the top of the file (`name`, `steps`,
`lanes`, `containers`), a step (`id`, `tool`, `config`, `inputs`, `lane`, `stage`, `container`,
`why`), a lane (`id`, `title`, `order`) and a container (`id`, `title`). A misspelt key would
otherwise be dropped without a word. The refusal names the key and suggests the one you probably
meant. `stage` is free text, so a misspelt stage is not caught: it draws as a stage of its own.

**A file written for an older version of the library** may use names that have since been renamed
(`groups`, a join's `matched` port, `strip()` and others). The loader refuses it, says what each old
name is now called, and names the command that rewrites the file:

```bash
dantron-etl migrate pipelines/          # a dry run: what would change, file by file
dantron-etl migrate --write pipelines/  # apply it
```

`python -m dantron_etl.migrate` is the same command.

It edits the file's text, so comments, layout and `why` text are kept exactly. Pass the rules files
too: an expression in one is migrated the same way. Python code is not rewritten; the command
prints the renamed Python names to change by hand.

**`why` is required to be useful.** It is the sentence a reader sees next to the number. Write what
the step decides and why that is the right decision, not what the configuration already says.

### 4.1 Ports

A step's inputs and outputs are **named frames**, not a single value. `inputs` wires one upstream
port to one of this tool's input ports:

```yaml fragment
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
| `join` | `left`, `right` | `joined`, `left_only`, `right_only` |
| `remove_duplicates` | `rows` | `unique`, `duplicate` |
| `expand_date_range` | `rows` | `rows`, `unexpanded` |
| `limit_rows` | `rows` | `rows`, `dropped` |
| `union` | `top`, `bottom` | `rows` |

`join` with `how: left` still writes to `joined` — `how` decides only which rows reach `joined`,
and the orphan ports answer their own question either way. So a left join reads `joined`; the rows
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
uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', concat([code], '|', [period]))
[placed_on] >= date'2025-01-01'
```

Available functions: `coalesce`, `round`, `year`, `month`, `day`, `quarter`, `day_of_week`,
`add_months`, `add_days`, `months_between`, `days_between`, `make_date`, `month_end`, `month_start`,
`parse_date`, `abs`, `floor`, `ceil`, `sign`, `mod`, `least`, `greatest`, `nullif`, `lower`,
`upper`, `trim`, `len`, `left`, `right`, `substr`, `split_part`, `pad_left`, `pad_right`,
`starts_with`, `ends_with`, `contains`, `regex_extract`, `regex_replace`, `regex_match`, `concat`,
`format_number`, `uuid5`.

**Dates.** `months_between(start, end, counting)` and `days_between(start, end, counting)` count
whole months or days between two dates, signed, with `counting` either `'inclusive'` (both ends
count, so the same day is 1) or `'elapsed'` (completed units only). They give exactly what the
`date_diff` tool gives, from the same code. `make_date(year, month, day)` builds a date from three
whole numbers and gives null when they name no date, such as 30 February; it is `assemble_date`
with `on_invalid: set_null`. `month_end(date)` is the last day of that date's month. All four take
dates; a timestamp in a time zone is refused, so cast it to `Date` first with `cast_columns` and
its `in_zone`, which chooses the day. Use
these instead of casting dates to whole numbers to subtract them: `days_between([from], [to],
'elapsed')` says what it counts, and a date cast to a number is a count of days since 1970 that a
reader has to know to trust.

`month`, `day`, `quarter` and `day_of_week` read a part of a date, as `year` does, and give a whole
number. `day_of_week` counts the ISO way: Monday is 1 and Sunday is 7. `add_days(date, count)` moves a
date by whole days, as `add_months` moves it by months; a count of more than 106,751 days stops the
step, because the engine cannot move a date that far exactly. A count that takes a date past what
the engine can hold stops `add_months` the same way. `month_start(date)` is the first day of
the month. `parse_date(text, format)` reads text as a date in a format written in the file, such as
`'%d/%m/%Y'`. A null stays null. Text that does not match the format, or names no real date, stops
the step and names the values: it never becomes a blank date. A format without a day, such as
`'%Y-%m'`, is refused when the file loads; write `parse_date(concat([period], '-01'), '%Y-%m-%d')`.

**Numbers.** `floor` and `ceil` go down or up to a whole number and keep the number's type, so a
`Decimal` stays a `Decimal`. `sign` gives -1, 0 or 1. `mod(a, b)` is the remainder with the sign of
`a`, as SQL's `MOD` and Alteryx's `Mod`: `mod(-7, 3)` is -1, where the engine's own `%` would give 2.
`mod` by zero stops the step, as `/` by zero does. On a `Decimal` it is exact, with the larger of the
two places: `mod(812.40, 3)` is 2.40. `least` and `greatest` take two or more values of one kind and skip nulls, as PostgreSQL does:
only all nulls give null. `nullif(a, b)` is null where `a` equals `b`, otherwise `a`;
`nullif([code], '')` turns empty text into a null.

**Text.** `left`, `right`, `substr`, `split_part`, `pad_left` and `pad_right` count characters, so an
accented letter or an emoji is one. Every count is a whole number written in the file, never a
column. `left([code], -2)` is all but the last two characters, as in SQL. `substr(text, start,
length)` counts `start` from 1; leave out `length` to run to the end. `split_part([ref], '-', 2)` is
the second part; past the last part it gives empty text, and a negative number counts from the
end. `pad_left([account], 6, '0')` pads to six characters with zeros and never cuts a longer value,
which is why it is not SQL's `lpad`, which does.

`trim` removes the spaces at the start and end of text. Unlike Excel's `TRIM`, it leaves the spaces
inside the text alone.

`format_number(value, decimals, thousands_separator, decimal_separator)` writes a number as text:
`format_number([total], 2, ',', '.')` gives `651,623.80` and `format_number([total], 3, '.', ',')`
gives `123.456,789`. All four arguments are required; the last three are written in the file, never
taken from columns. It gives the digits `round` gives (section 5). It works a row at a time, so use
it at the end of a flow, on the rows somebody reads.

`uuid5(namespace, value)` makes a stable 36-character id from text (RFC 4122 version 5), the same
on every run and every computer. Use it where the natural key is several columns.

- Build the value with `concat` and a visible separator:
  `uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', concat([code], '|', [period]))`. Without the
  separator, `('ab', 'c')` and `('a', 'bc')` get one id.
- The namespace is a UUID typed into the file, never a column. Make one per flow, once
  (`python -c "import uuid; print(uuid.uuid4())"`), and never change it: that changes every id.
- The value must be text. A number is refused, because `1` and `1.0` would be one value and two ids.
- It works a row at a time (about five microseconds a row). Derive the id once, where the key is
  built, not in every step that joins on it.

Operators: `and`, `or`, `not`, `+`, `-`, `*`, `/`, `=`, `!=`, `<`, `<=`, `>`, `>=`, `in`,
`is null`, `is not null`.

**Extend it by whitelisting a function, never by growing syntax.** There are no user-defined
functions, no attribute access, no subscripting, no lambdas. Those are refused by name, and that
refusal is a security boundary rather than a preference.

Loading checks an expression's syntax and the names it uses. Its types are checked when its step
runs, against the frames that actually arrived and before the step reads a row:
`upper([quantity])` on a number loads, and is refused when that step starts, not inside the query
engine. In a long flow that can be minutes into a run, so `until=` (§7b) is the quick way to try a
new step.

Some expressions that look reasonable never give the answer they seem to, and are refused:

- `[code] = null` is null on every row, so a filter keeps nothing. Write `[code] is null`.
- `regex_extract([ref], '[0-9]+', 1)` asks for group 1 of a pattern with no parentheses, which
  would be null on every row. Write `'([0-9]+)'`, or ask for group 0, the whole match.
- A timestamp compared with a date reads the date as midnight at its start, so
  `[posted_at] <= date'2025-01-31'` leaves out the 31st after midnight and `[posted_at] =
  date'2025-01-31'` is almost never true. `>=` and `<` mean what they say and are allowed; for the
  rest, cast the timestamp to `Date` first, or write the range as `>= start and < next day`.
- A `#` outside quotes is not a comment; put notes in the step's `why`.

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

A flow of any size is unreadable as a list of steps. `containers` names boxes and each step says which
box it is in:

```yaml fragment
containers:
  - {id: read_accounts, title: "Fetch the account list and put its header row back"}

steps:
  - id: accounts_raw
    tool: read_table
    lane: sources
    stage: parties
    container: read_accounts
```

On the drawn page a container is a titled box its steps sit inside, which a reader can close to a
single bar. That is the whole point: a box a reader can shut is a part of the flow they have decided
they do not need to look at today.

**What the loader refuses**, before a single step runs:

| | |
|---|---|
| a `container` naming no declared container | a typo is refused, never quietly ignored |
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
any step produces -- `container` is not part of a step's fingerprint (§7b), so boxing an existing flow
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

Each of these has caught a real pipeline.

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
- **`regex_replace`'s replacement is literal text, not a template.** `$1` is the two characters
  `$1`, not the first group, so a replacement of `$5.00` stays `$5.00`. To rearrange what a pattern
  matched, pull the pieces out with `regex_extract` and join them with `concat`.
- **`round` rounds a half away from zero, as a spreadsheet's ROUND does, on the number as you read
  it.** `round(2.5, 0)` is `3`, `round(-2.5, 0)` is `-3` and `round(0.125, 2)` is `0.13`. A double
  cannot hold `1.005` (it holds `1.00499999999999989...`), yet `round(1.005, 2)` is `1.01`: a
  decimal is read to 15 significant digits, the precision a spreadsheet works to, so a value that
  differs from a half only past its fifteenth digit is a half. That also catches arithmetic:
  `476.725` computed as `476.72499999999997` still rounds to `476.73`. When the digit you round on
  lies past the fifteenth significant digit, the value comes back unchanged. A whole-number column
  is unchanged; an exact decimal column rounds its own digits the same way.
- **`format_number` writes what `round` gives**, so the two never disagree: `-15.215` at two places
  is `-15.22` from both. A negative figure that rounds to zero keeps its minus sign, `-0.00`.
- **`uuid5` on a key with a missing part gives a null id, not an id.** A null anywhere in the value
  — straight from a column, or through the `concat` that builds the key — makes the whole id null,
  the way `concat` itself already behaves. The alternative is worse than it looks: hashing a
  stand-in would put *every* row missing that part under one real-looking id, which reads
  downstream as a legitimate key with several rows behind it rather than as data that was never
  there. A null id matches nothing and shows up as a null in a count, which is a question somebody
  asks. If a missing part should still produce an id, write the substitution where a reviewer sees
  it: `uuid5('...', concat(coalesce([code], 'none'), '|', [period]))`.
- **`format_number` writes a figure that is not a number as blank, not as a number.** Infinity and
  not-a-number can still arrive from a source, even though a division by zero inside a step now
  stops it (section 6). Formatting either gives null — an empty cell — because a figure that cannot
  be stated is stated as absent, the same answer a null gets. Use
  `coalesce(format_number([ratio], 2, ',', '.'), 'n/a')` if the report needs a word there instead.

### Columns

- **A blank name for an output column is refused by every tool that takes one.** Empty, or nothing
  but whitespace -- a non-breaking space pasted from a spreadsheet counts, and is the one you
  cannot see -- in `formula`'s `columns[].name`, `group_by`'s `aggregations[].name`, the `output`
  of `assemble_date`, `case_when`, `date_diff`, `normalize_text` and `record_id`, `unpivot`'s
  `names_to` and `values_to`, `expand_date_range`'s `period_column`, `rename_columns`' `to`,
  `make_groups`' `record_column` and `group_column`, and `regex_extract_groups`' `position_column`
  and `match_number_column`. A column called `   ` cannot be referenced later — `[   ]` is not something
  you can write — and it reads as an unnamed column in every rendering, so the step that wrote it
  reports success and the step after it cannot see what it produced. The rule is about a name that
  is *nothing*, not one that is unusual: `" total "` keeps its spaces and is written as spelled.
  The refusal is the same sentence whatever `replace` says, because a blank name is wrong either
  way.
- **`replace: true` on a column that is not there is a refusal, not a no-op.** Both `case_when`
  and `formula` take the flag, and it is a statement that the column is there to be written over. A
  misspelled name fails the run instead of quietly writing a new column and leaving the one you
  meant to repair with its old values. Without the flag, a name the input already carries is
  refused; with it, the column is written where it already was, and the cases read the values that
  were there before the step ran — so a case may read the column it writes. Reading the old values
  is not the same as keeping them — the column is written in full, so a row no rung reached takes
  `otherwise`, or null when there is none. A ladder that repairs three rows and has no `otherwise`
  nulls every other row of the column.
- **Every entry in one `formula` step reads the input as it arrived.** An entry cannot read a
  column an entry above it adds (that is refused: the column is not on the input), and an entry
  reading a column an entry above it *replaces* gets the old value. Replace `amount` with
  `[amount] * 2` and compute `total` as `[amount] + 1` in the same step, and 10 gives 20 and 11.
  This is SQL's `SELECT`, not Alteryx's Formula tool, which works top to bottom. `case_when` with
  `replace` reads the old values the same way. To read a value a step computes, use a second step.
- **A tool must not mutate its input.** The boundary checks each input frame's schema before and
  after a step and refuses a change, because the walkthrough shows every intermediate and an
  in-place edit makes an earlier step display columns it never produced.

### Types

- **The integer literal in an expression is 32-bit.** `0` and `1` in a `formula` give a narrower
  column than a database's `BIGINT`. It is the right width for a rank or a small count; use
  `cast_columns` when the width itself matters. A `sum` in `group_by` or `cross_tab` keeps the
  32-bit type, and a total too large for it stops the run. Arithmetic inside an expression is not
  checked that way: `year([d]) * 10000000` wraps round to a wrong number, so cast to `Int64` first.
- **`len()` gives an unsigned whole number.** Arithmetic on it, or on any unsigned column, is done
  as a signed 64-bit number, so `len([code]) - 5` is -2 for a three-letter code rather than wrapping
  to about four billion.
- **A decimal becomes a whole number only when it already is one.** `cast_columns` (and `unpivot`'s
  `values_type`) reads `12.0` as `Int64` 12, but stops on `1.99`, naming the column and the values,
  because the engine would cut it to `1` without a word. Round it first: `round([seats], 0)`.
- **`Decimal` is the exact type, and its places are written beside it.** `{type: Decimal, places: 2}`
  names a column that holds figures such as 812.40 exactly: 38 digits in all, `places` of them after
  the point, from 0 to 38. No tool takes it yet. `cast_columns`, `read_csv`, `unpivot` and
  `write_excel`'s rules refuse it by name; use `Float64` until they do.
- **Arithmetic on an exact column is exact.** `+` and `-` give the larger of the two places:
  `[amount] + 0.0725` on 812.40 is 812.4725. A number typed in the expression is exact as written,
  so `1.10` has 2 places, and a whole number counts as 0 places. An answer that needs more than 38
  digits stops the step and names the rows; it never rounds or wraps round.
- **`*` adds the two places, and never rounds.** 812.40 × 0.0725 is 58.899000, six places. Places
  grow quickly in a chain of products, and a product whose places pass 38 is refused when the step
  starts. Round one side first, such as `round([rate], 4)`, when a figure only needs so many.
- **A calculation of typed numbers alone is exact beside an exact column:** `[amount] * (1 + 0.0725)`
  multiplies by exactly 1.0725, and `1 / 3` there is 0.3333333333333333 by the division rule.
- **`/` gives 16 places, or more if either side has more,** and rounds the last place half away from
  zero: 100.00 / 7.00 is 14.2857142857142857. Round the answer to the places you report. At 16 places
  each side may have at most 22 whole digits; a larger one stops the step.
- **`round(x, n)` on an exact value gives exactly `n` places**, half away from zero: `round(58.899000,
  2)` is 58.90 and `round(1.5, 2)` is 1.50. `n` runs to 38. Apart from the last place of a division, it is
  the only place an exact figure is rounded.
- **An exact value never meets a float.** `[amount] + [ratio]`, `[amount] = [ratio]`,
  `coalesce([amount], [ratio])` and every other function that combines values stop the step when
  `amount` is `Decimal` and `ratio` is `Float64`, naming the float. Convert one side first with
  `cast_columns`. A whole number or a typed number beside an exact value is exact:
  `coalesce([amount], 0.0)` gives 0.00 where `amount` is blank.
- **Exact values compare by value.** 1.50 equals 1.5000, and `[amount] in (1.5, 812.4)` matches
  1.50 and 812.40. Nothing is rounded to match: 0.12 is not in `(0.125)`.
- **A date cast to a number is a count, and its unit depends on the type.** `cast_columns` turns a
  `Date` into the days since 1 January 1970, and a `Datetime` into a count in the column's own time
  unit (microseconds for one this library typed; a source may use nanoseconds). To count the days
  or months between two dates, use `days_between` / `months_between` or the `date_diff` tool, which
  say which they count.
- **A format rule for `Int64` reaches every whole-number column**, including the `Int32` column an
  integer literal gives. A rule for `Float64` does not reach a `Float32` column: cast it to
  `Float64`, or name the column in the rule.
- **Which day a timestamp in a time zone falls on depends on the zone, so you state it.** 23:30 on
  31 January in Toronto is 1 February in UTC. `cast_columns` to `Date`, `date_diff` and
  `expand_date_range` refuse such a timestamp unless `in_zone` says which zone's calendar to use:
  `in_zone: UTC` reads the UTC date, which is what they read before the rule. There is no default,
  and `in_zone` where no timestamp carries a zone is refused too.
- **A spreadsheet tab refuses not-a-number and infinity.** `write_excel` would show them as blank
  cells, which read as "no figure". It stops instead, naming the tab, the column and the rows. A
  null is written as an empty cell, as before.
- **A spreadsheet cell cannot record a time zone**, so `write_excel` refuses a
  zone-aware timestamp. `cast_columns` it to `Date` first, with `in_zone`, deliberately, rather
  than letting a writer drop the zone quietly.
- **A text file states no types at all.** `read_csv` infers, reading the *whole* column before it
  casts. Declare `column_types` as `String` for any column that is an identifier or a name: a
  customer literally called `3600` inferred as a number stops matching the same name everywhere
  else, and nothing about that failure looks like a failure. A column of numbers that a number
  would change stops the read instead of being guessed: `02134` would lose its zero, and a whole
  number past 64 bits would be rounded. Declare it `String` to keep it as written.
- **A workbook column takes its type from its first 100 rows.** `read_excel` stops when a cell
  holds something the column's type would turn into null: text far down a column of figures, or a
  word such as `n/a` or `#N/A` anywhere in one. Set `as_text: true` and convert with `cast_columns`.
  A cell holding an error value that a formula left behind (`#N/A`, `#DIV/0!`) is the exception: it
  reads as null and is not refused. Check for nulls with `assert_rows` where that matters.
- **`read_excel` reads `.xlsx` and `.xls`**, Excel's older format, by the same rules: the same
  cells give the same rows, the same types and the same refusals, whichever day an `.xls` counts its
  dates from. A file with an `.xls` name that holds a web page or delimited text is refused, saying
  so; read delimited text with `read_csv`.

### How a figure is shown

- **Nothing is formatted unless the tab says so.** `write_excel` writes the figures and leaves
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
- **`first` means "the first row as it arrives", null or not.** It is only meaningful after a
  `sort`. A `first` with no ordering in front of it is a coin toss that will land the same way in
  testing and differently in production. A group whose first row is blank gives null even when a
  later row has a value; `first_non_null` gives the first value that is there instead. `last` and
  `last_non_null` are the same from the other end.
- **A database hands rows over in no fixed order.** `read_table` keeps whatever order the database
  sent, and the database promises none. So `remove_duplicates` with `keep`, `group_by`'s `first`
  and `join_text`, `record_id` and `limit_rows` read an order nobody chose when they sit after a
  `read_table` with no `sort` in between. Put a `sort` there.

### Paths

- **A relative `path:` is read from the project folder when the `dantron-etl` command runs the
  pipeline**, wherever the command was started. From Python it is read from the working directory
  of the process, not from the pipeline file's folder; pass
  `run_pipeline(spec, base_dir=project_folder)` to read it from a folder you name.
- **A writer stops on an empty frame unless the step allows it.** `write_table`, `publish_tables`
  (per table), `write_csv` and `write_excel` (per sheet) all refuse a frame with no rows until
  `allow_empty: true` says the branch may be empty, because a table or file that is empty for a
  reason upstream looks exactly like a quiet month. Allowed, a file is written with its header and
  a tab with its headings.
- **A file writer is checked against the disk when the pipeline loads.** The folder of its `path`
  must already exist. A file already at `path` is refused unless the step says it may be replaced,
  so a flow that runs every night needs `overwrite: true` on `write_csv` and `write_excel` from its
  first night, or its second run will not load. On `write_excel` the same word also lets a sheet
  entry replace a tab of the template.
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
- **A `publish_tables` merge says what a null in tonight's row does.** `when_incoming_is_null:
  keep_published` leaves the published value standing, so one unreadable source cannot blank a
  column. `overwrite` writes the null, as SQL's `MERGE` does. There is no default: the two publish
  different figures.
- **`count` counts rows. `count_non_null` counts the values that are there.** Other libraries spell
  the non-null count `count`, so a step ported from one silently returns the other number.
- **`mean`, `median`, `std` and `quantile` skip nulls**, as SQL's `AVG` does, so a mean divides by
  the values that are there. A group with no value gives null. They refuse an exact `Decimal`
  column, which they would work out in binary floating point; cast it to `Float64` first.
  `quantile` needs `fraction`, from 0 to 1, beside it. `any_true` and `all_true` skip nulls too, and
  a group of nulls gives null, as SQL's `BOOL_OR` and `BOOL_AND`.
- **`count_distinct` skips nulls, as SQL's `COUNT(DISTINCT)` does.** `a, a, null, b` counts 2, and a
  group of nulls counts 0. To count a blank as a value too, count `coalesce([v], '(blank)')`. It
  was called `n_unique` and counted a null as one more value; the migration renames it.
- **A division by zero stops the step.** It has no answer, and the infinity or not-a-number the
  engine would give reads as larger than every number in a later `filter` or `sort`. So a `/`
  whose divisor is zero (`0`, `0.0` or `-0.0`) on any row the step uses stops the run, naming the
  step, the field, the expression and up to five of the rows; a divisor written as zero is refused
  when the file loads. A null divisor is not zero: the answer is null. Not-a-number is not zero
  either: dividing by it gives not-a-number, as before. A flow that expects zeros guards them in
  `case_when`, with a rung of their own before the rung that divides: `case_when` checks each
  division only on the rows that reach it, as SQL's `CASE` does. A guard inside one expression,
  `[b] != 0 and [a] / [b] > 1`, does not count, because every part of an expression is worked out
  on every row.
- **Not-a-number from a source is not a null.** It compares as larger than every number and equal
  to itself. So `[ratio] > 0.5` is true for it in `filter`, `case_when` and `assert_rows`, and
  `sort` puts it after every number ascending and before them descending, while `nulls` moves only
  the nulls.

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

```python fragment
trace = run_pipeline(spec, until="clients_joined")
```

What runs is the *ancestor closure* -- every step the named one depends on, however far back, and no
step that merely happens to be written earlier in the file. Naming a step the pipeline does not have
is refused before anything runs.

**`cached`** hands a port back to a later run so the steps behind it do not run at all:

```python
from dantron_etl import CachedFrame, PortRef, read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
first = run_pipeline(spec, until="clients_joined")
port = PortRef("clients_joined", "rows")

later = run_pipeline(
    spec,
    cached={port: CachedFrame(first.frame(port), spec.fingerprint(port))},
)
```

`CachedFrame` carries the frame **and** a fingerprint, and neither is any use alone. The fingerprint
is `spec.fingerprint(port)`, a hash over the step that produced the port and every step behind it --
their ids, tools, configurations and wiring -- and over the port's name, so the `true` frame of a
split cannot be resumed as its `false` one. For a step with one output port, `spec.fingerprint(step_id)`
gives the same string. Before the run walks a cached port it recomputes that hash and compares:
if anything upstream has been edited since, the cache is **refused by name** rather than used.

That refusal is the whole feature. A cache that fails to load is a nuisance; a cache that loads
perfectly against a pipeline somebody has since edited is a wrong number nobody can see.

Deliberately **not** part of the fingerprint: `lane`, `stage`, `why` and `container`. None of them
changes a figure, so re-drawing or re-describing a flow does not throw away a cache you would still
have been right to use.

**The loop this is for:** cache the port just before the part you are editing, then run only from
there while you iterate. Run whole once at the end -- a cached run proves what your edit did, never
that the flow as a whole still agrees with its oracle.

**`retain` and `keep`** decide which frames a run holds when it ends. The default, `retain="all"`,
holds every one. `retain="live"` frees each frame as soon as nothing left in the run reads it, which
is what a scheduled run of a large flow wants. `keep` names the ports to hold on to anyway:

```python
from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
trace = run_pipeline(spec, retain="live", keep=["published.receipt"])
```

### The one thing a fingerprint cannot tell you

It is taken over **the file**, never over the data. `id`, `tool`, `config` and `inputs` -- nothing
else. So a cache survives a change to what the sources hold, and survives it silently:

> Cache a port. Reload the tables underneath. Resume from the cache. The steps behind that port do
> not run, so the new rows are never read, and the run reports a clean resume over yesterday's
> figures.

Nothing catches this, and nothing can: the reads did not happen, so there is no new frame for
anything to compare against. **A cache is only valid for the data that was there when it was
taken.** Throw one away whenever the sources are reloaded, and take a fresh one.

The fingerprint does not cover the library either: a cache taken before you upgrade the library, or
edit it in place, still matches afterwards even when a tool now computes something different. Take
a fresh cache after changing the library.

### What the viewer keeps of a run

The viewer and `dantron-etl run` save the rows at every port of each run in the project's runs
folder (`runs_folder` in `etl-project.yaml`, by default `output_files/runs`), so the page can show
them. Those rows can be personal or financial data.

- On Linux and macOS, each folder a run makes is for its owner only (`0700`), and so is each file
  in it (`0600`), whatever the umask. Folders that were already there keep their permissions, and so
  do older runs and the files your output steps write.
- On Windows, and on a drive that refuses the change, the permissions are not changed. The run is
  still saved, and its files are as private as the folder they sit in. Keep the runs folder
  somewhere only you can read.
- `save_rows: false` in `etl-project.yaml`, at the top or under one pipeline, keeps no rows at all:
  the run keeps its steps, timings, row counts and the columns and types at every port, and the
  viewer says where rows would be that they are not saved. A failed step's message is kept, since it
  says why the run stopped, and can name one value; the example rows it lists are left out.

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

Open the issue or pull request against the repository this package is distributed from: the
`Source` URL in the distribution's metadata (`project.urls` in its `pyproject.toml`). Its
`CONTRIBUTING.md` lists how to set up, the tests to run and the standards a change is held to.

---

## 9. A worked shape, end to end

A common pattern: reduce a group to one row, keep the detail that would otherwise be lost, and check
the result. These steps continue the file in section 4.

```yaml fragment
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
