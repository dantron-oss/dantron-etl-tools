# Tools: design notes

Why each tool behaves as it does. The code says what it does; these notes keep the decisions behind
it, the alternatives that lost, and what was measured. One section per tool that has a ruling worth
keeping. The general rule they all share: **each tool follows the SQL operation it is named after**,
and that is how its handling of nulls is decided.

## join

**Three output ports instead of a `how` with many options.** One pairing gives `joined`, `left_only`
and `right_only`. A right join is the same join with the inputs swapped, an outer join reads all
three ports, and an anti join reads `left_only`, so none of those is an option. Two spellings of the
same result would be one more thing to get wrong. The unpaired rows become an edge a flow can send to
a report or an assertion, instead of a second join someone has to remember to write. `how: left`
was added because a left join is the join a flow asks for most, and building it from two steps was
the commonest piece of boilerplate.

**Fan-out is kept and shown.** A key that appears twice on the right turns one left row into two, and
an amount on that row is repeated, not split. Picking one match would need a rule the configuration
does not give. Keeping the key unique is the caller's job (remove_duplicates or assert_rows before
the join). The repetition stays visible so a reader sees it before trusting a total.

**A null key never pairs.** This follows SQL's `=`: a missing key is missing information, and pairing
two of them would invent a relationship. On dirty data it would also pair every null-keyed row on one
side with every null-keyed row on the other, turning a small gap into a large wrong number.
remove_duplicates decides the other way because it follows `DISTINCT`, where two nulls are one group.
Both are right for the operation they follow.

**Row accounting is about rows, not counts.** With fan-out, `joined.height + left_only.height ==
left.height` is false. The promise is that every row of a side is either on `joined` at least once or
on its own unpaired port exactly once, never both and never neither (under `how: inner`).

**Row order is fixed.** `joined` comes in left order, with a left row's copies in right order; each
unpaired port keeps its own side's order. This costs polars' `maintain_order`, and the parallelism it
gives up. It is paid so a flow that writes a file gets a diff that does not churn between runs. It is
not a promise to rely on for correctness: a step that needs an order still needs a sort.

**The unpaired ports are anti joins.** They are not a left join filtered on a null right-hand column,
because a paired row can hold a null in an ordinary column and would be miscounted as unpaired. Only
the other side's key columns are read.

**Clashing names are refused from the schema.** When the suffixed name of a right-hand column is
also taken, the step is refused before any row moves, naming `suffix` as the fix. Left to polars, it
fails later with a duplicate-column error that names neither the setting nor the fix.

**Key types are checked by asking polars.** Two empty one-column frames are joined to see whether
the types compare, so the check accepts exactly what a join accepts (Int32 with Int64 yes, Int64 with
Float64 no). A key column typed Null, as an empty branch gives, takes the other side's type so
`joined` keeps its usual columns; none of its rows pair.

**Exact keys are compared by value, by the tool.** polars joins only keys of one type, so it refuses
`Decimal(38,2)` against `Decimal(38,4)` and against a whole number. SQL's `=` compares both by value,
so the tool gives each side a copy of the key at the larger places, joins on the copies and drops
them; every original column keeps its own type. A value with too many digits before the point to be
held at the larger places becomes null in its copy and pairs nothing, which is the right answer: no
value on the other side can equal it. A float key against an exact key is refused (the owner's ruling
that an exact column meeting a float stops the step), naming the float column.

## group_by

**Aggregates are settings, not expression functions.** The expression language works one row at a
time and does not list `sum`. Allowing `sum([amount])` in one setting only would make a second
dialect a reader has to learn from where a string sits. So group_by has its own short list,
`AGGREGATE_FUNCTIONS`, and the `function` setting's allowed values are built from it, so the two
cannot drift. cross_tab imports the same list, so adding an aggregate adds it to both.

**`mean`, `median`, `std` and `quantile` skip nulls**, as SQL's `AVG`, `PERCENTILE_CONT` and
`STDDEV` do, so a mean divides by the values that are there, not by the rows. `std` is the sample
deviation (`STDDEV`, Excel's `STDEV.S`), so one value has none. `quantile` interpolates
(`PERCENTILE_CONT`, Excel's `PERCENTILE.INC`) and needs `fraction`, placed beside it as `separator`
sits beside `join_text`. On an exact `Decimal` the engine answers in binary floating point (the
median of 0.20 and 0.70 comes back as 0.44999999999999996), so a `Decimal` column is refused and the
cast to `Float64` is written in the pipeline. `any_true` and `all_true` are `BOOL_OR` and `BOOL_AND`:
the engine answers false and true for a group of nulls, so that case is guarded to give null.
These eight are worked out by one function, `shared_aggregate`, that cross_tab calls too.

**Groups keep first-appearance order.** A key of several columns has no obvious sort order, and the
order the input already has is the one a reader knows. polars promises no group order unless asked,
so `maintain_order=True` is set.

**A null key is a group.** This follows SQL's `GROUP BY` and agrees with remove_duplicates. Dropping
those rows would delete their amounts from every later total with nothing left to say they were
there. join decides the other way because it follows `=`.

**A sum of nothing is null.** polars sums an all-null group to 0. Passed on, that is a figure nobody
measured, and nothing later can tell it from a real zero. Nulls are still skipped in a group with
any value, so all-null is the only case that gives null.

**`first`, `last` and `join_text` read arrival order.** `first` exists for a column that is the same on every
row of a group (a document number on its lines), where any row's value is all of them. When the
column does vary, `first` answers with whichever row came first, a fact about order, not data. A
flow that needs a repeatable answer sorts first, and can add `count_distinct` beside it to check the
column really is constant. Owner's ruling (2026-11-18): `first` keeps the first row's value even when
it is null; `first_non_null` was added for the other reading. `last` and `last_non_null` mirror them.

**`count_distinct` skips nulls**, as SQL's `COUNT(DISTINCT v)` does (owner's ruling 2026-11-18). To
count a blank as a value, count `coalesce([v], '(blank)')`.

**`count` and `count_non_null` are two names, not one name and a flag.** `count` counts rows and takes
no `value`; `count_non_null` needs one. pandas spells the non-null count `count`, so someone porting
a step would otherwise get the other number without being told.

**`join_text` is SQL's `string_agg`.** Values are joined in arrival order with `separator` (default
`", "`). Nulls are skipped, so `a, null, b` gives `"a, b"`, not `"a, , b"`, which would read as a blank
value instead of a missing one. A group with nothing to join gives the empty string, the opposite of
`sum`: joining no strings gives an empty string in every language, while adding no numbers gives a
total nobody measured. A `separator` beside another aggregate is refused, not ignored, so `min`
written for `join_text` does not quietly give single values. Only text is joined: a number joined
into a string becomes a string of digits that can no longer be sorted or added.

**A value that reads no column is refused.** polars works out a fixed value once per group, so
`sum` of 1 gives 1 for every group, not a row count.

**Narrow integer sums do not wrap.** polars sums Int32 and UInt32 into the same type and wraps round
past the limit without a word. Those sums are worked out as Int64, checked against the type's limits,
and cast back, so the column keeps its type and a total that does not fit stops the run. Smaller
integer types already sum into Int64.

**Every aggregate has its own branch, and an unknown name raises.** A dispatcher that ended by
returning one aggregate once computed cross_tab's `first` as `max`, a plausible number under the
wrong name.

## write_excel

**One step writes many tabs, and its input ports come from `sheets`.** Each sheet entry names a port,
so the step reads exactly the ports its configuration lists (owner's approval, 2026-09-11: "I like
that it declares its own number of inputs"). The alternatives lost: a fixed set of ports
(`sheet_1` to `sheet_20`) caps the count and reads badly; one stacked input with a column naming the
tab needs every tab to share columns, and a real pack's tabs ran from 1 column to 49. The tool
implements `input_ports_for` and leaves `input_ports` empty; a tool does one or the other.

**One load and one save.** Measured on a real pack of 12 tabs (0.56 MB):

| approach | time | keeps the template |
|---|---|---|
| polars into one open xlsxwriter workbook | 1.46 s | no, cannot start from a template |
| openpyxl, one load and one save | 2.68 s | yes |
| openpyxl, one separate write per tab | 34.44 s | yes |

openpyxl with one load and one save is the only fast option that keeps a template. That is also why
the one refusal that needs the workbook open (a template tab named without `overwrite`) is made in
`execute`, before the save: asking earlier would load a file of several megabytes twice.

**No output port.** Handing the frames back out would invent a fan-out, and a receipt frame would be
shape without information. A flow ends here.

**The template is a curated workbook.** A delivered pack is usually built by hand: charts, formats,
notes. That workbook is the template, and only the tabs whose figures change each run are written.
The template is read and never written; a `path` equal to the template is refused, because a
template overwritten by its first run leaves the next run nothing to start from. Sameness is checked
twice: resolved paths first (catches `..`, relative paths, symlinks, and works when `path` does not
exist yet), then `samefile` when both exist (catches hard links). A link made between the check and
the save is not seen, and a mount that gives every file the same identity can cause a false refusal.

**Rewritten, not copied.** openpyxl reads the template into memory and writes a new file. Measured
coming through: tabs, formats, formulas, charts, pivot tables and the cached values behind a pivot's
filter dropdowns. Lost: each formula's stored result (the file recalculates when a spreadsheet opens
it, but a program reading it before then sees blanks), and a pivot cache's extension block, which
holds newer pivot features such as a slicer link. A test records that loss.

**The file at `path` is never read.** A tab the flow stopped writing once lingered in the pack because
the writer wrote into the file already there. Each run now builds from the template or from nothing,
so the result is the same whatever an earlier run left. `overwrite` is the single permission to
replace what is there: the file at `path` (refused at load) and a template tab a sheet entry names
(refused when the template is opened). A replaced tab is remade in its old position, because a tab
that moves to the end gets reported as missing.

**An empty port stops the run unless `allow_empty`.** An empty tab caused by an upstream failure
looks like a quiet month. Owner's ruling (2026-09-11): when allowed, a port with no rows writes a tab
with its headings, never no tab. A missing tab reads as a broken report; an empty one reads as
"nothing this month". No placeholder phrase such as "(no rows)" is written, because a reader's
formula expects a heading there.

**Formats are stated, never guessed.** A date column gets a date format, because without one a cell
shows a count of days since 1899. Nothing else is formatted unless a rule says so. Deciding that a
decimal column is money is wrong the first time someone puts a reference number in one; deciding from
the heading is wrong the first time it is renamed.

**A rule names a type or one column.** A type rule reaches columns the configuration could not name,
such as the one-per-month columns of a cross_tab; naming them one by one means editing the flow every
month, and the forgotten month is the one unformatted column. A column rule beats a type rule
whatever the order, so the general rule can be written first and the exception under it. A repeated
type or column is refused: picking the first or the last is a guess. An Int64 rule also reaches Int32,
because a formula makes a whole-number literal 32-bit.

**Format codes are passed through unchecked.** The library cannot know what a future spreadsheet
accepts; a code wrongly rejected would block a pack for nothing, and a wrong code shows visibly in the
file instead of changing a figure. A rule about Time is accepted and does nothing, since a Time column
is refused anyway; refusing the rule would tie two lists together for no gain.

**Formats go on data cells, not whole columns.** A column-wide format would run to the bottom of the
tab and format a figure typed in below the data. Headings never get a number format. An empty cell
is formatted like its column, so typing into a gap does not reveal a chequerboard.

**What a cell cannot hold is refused, not converted.** Rules shared by every workbook writer live in
`_spreadsheet.py`. A whole number past 2**53 - 1 (the largest a double holds exactly, with its
neighbour still distinct) is refused, not rounded; cast it upstream where the next reader can see it.
Time, Duration, Decimal, lists, arrays, binary, nested records and zone-aware timestamps are refused:
each has a plausible-looking wrong answer in a cell. Text over 32,767 characters would be cut short;
more than 1,048,575 data rows would be truncated, the worst outcome for a delivered total.
Not-a-number and infinity would show as a blank, as if no figure were there. Every frame is checked
before the workbook opens, so a refusal about the last tab leaves no half-written pack.

**Text starting with `=` stays text.** openpyxl stores it as a formula, so `=HYPERLINK(...)` in free
text would run on the reader's machine. The cell is forced back to text with a quote prefix.

**Layout never touches a figure.** `widths`, `fit_widths`, `wrap`, `table` and `header` run after the
cells are written and only set fonts, widths, alignment and tables; a test fails if any of them
changes a value or a number format. `fit_widths` measures a value as shown (1234567.0 under
`#,##0.00` is `1,234,567.00`), because a column fitted to the stored digits shows `####`. A fitted
column is capped at 60 characters so a paragraph does not make a column hundreds wide; a stated width
goes past it. Headings are measured at their font size, plus room for a table's filter button.

**Tables.** A table's name is the one given, or the tab name with unusable characters turned into
underscores. Two tables of one name are refused ignoring case, as a spreadsheet compares them, and so
is a name the template already uses on a tab this step leaves alone. A port with no rows gets a
table over its headings and one empty row, the smallest a spreadsheet accepts.

**Tab names are refused, not repaired.** People write dates with slashes and headings with colons
into tab names. Renaming quietly delivers a tab nobody can find under the name the flow uses.

## filter

**Two output ports, nothing dropped.** A filter that kept only matching rows would turn the rest into
something a reader rebuilds later with an anti join, if they remember. Every row goes to exactly one
of `true` or `false`, so `true.height + false.height == rows.height` always.

**A null condition goes to `false` by default.** A comparison with a null is null, as in SQL, and
SQL's WHERE drops it. `nulls: include` sends it to `true`. There is no third port for null rows: the
split is one step away (`[column] is null` on `false`), and a third output would put an almost
always empty edge on every flow.

**The setting values are `exclude` and `include`, not `false` and `true`.** YAML turns an unquoted
`true` into a boolean, so `nulls: true` would be refused with a type error unrelated to what was meant.

**A condition must give true or false, checked on its type.** `[amount]` on a number column is a
valid expression but not a condition; polars would fail with a type error from a library the author
never heard of. A bare true/false column is a valid condition, because filtering on a status flag is
a real use.

## limit_rows

**Two output ports.** A tool that quietly discards rows is how a report shows the top 500 of
something while reading as if it showed everything. The rest leave by `dropped`, with the input's
columns exactly. Put back together, the two ports are the input, in order.

**It never sorts.** "Top N" elsewhere usually means a sort then a limit, built into one tool. Here
the sort is its own step in front, where its options are written down.

**`count: 0` is allowed.** Keeping none is a real setting, and a count that comes from a parameter
should behave like a limit when it reaches zero, not stop the run. Nothing is lost: every row goes
to `dropped`. A negative count is refused, quoting the number (to expose an off-by-one upstream) and
naming 0 as the way to keep none.

**`count` is a whole number with no default.** A plain number would accept 2.5 and leave the tool to
round it, a policy nobody wrote down. A default of, say, 1,000 would truncate a frame silently, the
very failure the second port exists to prevent.

## select_columns

**Its own step.** Carrying every column a flow ever read makes every later step harder to read.
Narrowing to what is needed, or dropping what has served its purpose, is common enough to be a step
of its own. Both directions share one setting list and an `action`.

**Order follows the list under `keep`.** The list doubles as a reordering. Under `drop` there is no
list of survivors to follow, so they keep the frame's order.

**An entry that matches nothing is refused, wildcard included.** A wildcard that silently added zero
columns would be a whitelist failing open: the schema moved under a configuration written for an
older one, and the author needs to hear which entry stopped matching.

**Only `*` is a wildcard; matching is case sensitive.** A header copied from a spreadsheet, such as
`Total [CAD]`, must select itself and nothing else, so brackets and `?` are literal. An entry equal to
a column's name selects that column even if it contains `*`.

## rename_columns

**Why it exists.** Without it, a rename is a formula that copies the column under a new name plus a
select that drops the old one. The copy is appended, so the column moves to the end of the frame,
and anything reading columns by position gets a different frame without being told. Keeping the
renamed column where it was is the reason for the tool.

**Renames are applied together.** The list is one mapping over the input's columns, not a sequence,
so `code -> region` beside `region -> code` swaps the names, and `code -> region` beside
`region -> territory` needs no temporary name. Applied one at a time, the pairs would either fail
with a duplicate-column error from polars or end with two columns under one name.

**A collision is refused, but a name freed in the same step is not a collision.** Writing over a
column that stays loses it silently, so it is refused by name. The check must still allow the swap.

**A rename that changes nothing is refused.** A rename to the same name, a blank name, a name of
only spaces: each reads as an instruction and is not one. The usual cause is a configuration edited
halfway, and saying so beats agreeing quietly. Every refusal comes from the configuration and the
column names, never from a row, so an empty frame is refused the same way.

## sort

**Each key has its own direction and null placement.** One list can say "zone ascending with nulls
first, then amount descending with nulls last". A tool that applied one direction to every key would
look right on the single-key configuration everyone writes first and be wrong on the second key of
the next one.

**The sort is stable, and enforced.** Rows that tie on every key keep their arrival order. Other
tools build on that: record_id numbers rows after a sort, and an unstable sort would make the numbers
depend on how polars laid out the rows, so two runs could differ. polars' own `maintain_order` does
not hold in every case: when every key is null on every row and a key sorts descending, polars 1.38
reorders the rows (with one key, in reverse). So each row's arrival position is added as the last
key, ascending. The flag is kept too; it costs nothing and says what is meant.

**A setting written with no value is refused.** `direction:` with nothing after it arrives as null.
Treated as "not set", it would sort by a policy the author never chose.

**Defaults are applied by the tool.** The config template fills defaults only at the top level, so a
default declared inside a sort key would look as if it applied and never would.

**A repeated key is refused.** The second one can never decide anything: the first already ordered
every row it would look at.

## union

**Why it exists.** filter, remove_duplicates and join split a frame into ports; none puts two frames
back together. Without union, a flow that treats rows apart cannot carry them on as one, and a flow
reading the same thing from two sources runs every later step twice. The name stays `union` (owner,
stage 4) because it is Alteryx's word for stacking; the help says plainly that it keeps duplicates and
matches columns by name.

**A missing column is refused, not filled.** Filling it with nulls is the quiet version of a join that
lost half its rows: the frame has the expected shape and a column of nothing where the data was. A
flow that wants the column adds it with a formula, where it is written down.

**Column order may differ.** The same columns in a different order are the same data. Refusing them
would force a select_columns step whose only job is to please this tool.

**Types must agree, except an all-null column.** Text stacked onto a number turns money into text, and
whichever side was rewritten, it happened silently; the refusal names both types. A column of nothing
has no type to disagree with: null belongs to every type, and an empty branch is an ordinary frame.

**Exact columns of different places widen.** 1.5 at 1 place and 1.50 at 2 are the same value, and
adding zeros loses nothing, so they stack at the larger places. The tool widens them itself: the
engine's own stacking keeps the first side's places and rounds the second (1.2345 under 2 places
becomes 1.23). A value with so many digits before the point that the larger places cannot hold it is
refused by name. An exact column over a whole number stays refused, as Int32 over Int64 is: union
refuses every other type difference.

**Rows keep their order and nothing is de-duplicated.** All of `top`, then all of `bottom`. Stacking
is not judging; remove_duplicates reports what it removes.

## remove_duplicates

**Two output ports.** A step that only kept survivors would make the removed rows stop being data,
a count someone rebuilds later if they remember. So every row goes to `unique` or `duplicate`, and
`unique.height + duplicate.height == rows.height` always, the same rule filter follows.

**Sameness follows DISTINCT, not `=`.** Rows whose key is null are one group, matching polars and what
a later group_by would do. Reporting each null-keyed row as unique would leave exactly the rows this
tool exists to clean. join decides the other way because it follows `=`. A null is still not the empty
string: a text key made by filling nulls first would merge real blanks into the null group.

**No `by` means the whole row.** An empty list means the same. There is no `["all"]` placeholder; one
carried over from an older tool is refused as a missing column, by name.

**`keep` is `first` or `last`,** never a word YAML would turn into a boolean.

**Order is kept within each port.** polars' `unique` does not promise order, so the tool uses a per-row
mask (`is_first_distinct` / `is_last_distinct`) filtered in place, which keeps order and gives both
ports from one pass.

## record_id

**The numbering follows arrival order.** The tool sorts nothing and decides nothing, which is why it
is worth something only after a sort step: then the numbers follow an order someone wrote down. The
numbers are built from the frame's height alone, never from the data.

**The column is declared Int64.** A frame with no rows has nothing to infer a type from, and a column
typed Null would break the next join on the day an upstream filter matched nothing.

**A name already in use is refused.** The author of the step may not know every column of the frame;
a collision taken quietly destroys a column and reports success. There is no overwrite option.

**Overflow is refused twice.** A start outside Int64 is wrong whatever the frame, so it is refused
from the configuration. A start that fits but would pass the ceiling part way down depends on the row
count, so it is refused when the frame arrives; left alone it would wrap to large negative numbers.

## formula

**A list of `{name, value, replace}` entries, not one `[out] = expr` string.** A single string would
pack the output name, the expression and the number of columns into one opaque value. A real step
often writes several columns, so the setting is an ordered list.

**Every expression reads the input as it arrived.** This matches polars' `with_columns`, which works
out every expression against the frame before any of them ran. A second value that needs the first
goes in a second formula step, which shows the dependency in the flow instead of hiding it in the
order of a list. One `with_columns` call also keeps a replaced column in its place: polars keeps a
name already in the frame where it was and appends only new names. A client flow depends on this
behaviour (finding E-11), so it is kept and documented.

**Division by zero stops the run** (owner's ruling 2026-11-18, finding C-02). Left to polars it gives
infinity, or not-a-number for 0/0, which a later filter reads as larger than every number. The step
names the field, the expression, the divisor and some of the rows. A null divisor is not zero: the
division gives null. A flow that expects zeros guards with case_when first.

## case_when

**Order decides.** A single condition is formula's job. case_when exists for several cases that can
overlap, such as "superseded" checked before "amount is missing" though a superseded row may also lack
an amount. Cases are read in the order written and the first one a row meets wins. Nothing rewrites
that into an order-free form: an overlapping list has exactly one correct reading, the author's.

**A null condition falls through.** A comparison with a missing value is null, not false, as in SQL.
Such a row is not a match for that case and moves on, ending at `otherwise` or null.

**Every value must be the same kind.** polars would turn a number or a true/false into text the moment
it sits beside text, silently, giving a column of plausible strings. So text, number, date and
true/false must agree, checked before any row runs and naming the branch. Null fits every kind. Whole
and decimal numbers count as one kind because polars mixes them correctly.

**`replace` follows formula.** Without it, a name already on the frame is refused: overwriting a
column a later step reads is a wrong answer with no failure. With it and no such column, it is refused
too: the flag claims a column to repair, and a misspelling would write a new one. A replaced column
stays in place, and the cases read its old values, which is what blank-to-null and status-collapsing
steps need. Every row is rewritten, so a row no case reached gets `otherwise` or null.

**Division by zero is checked where it is used** (owner's ruling C-02, applied as SQL's CASE). A case's
condition is checked on rows no earlier case took, its value on the rows it takes, `otherwise` on rows
no case took. So a case that catches the zero rows first guards a later division.

## assert_rows

**One output port.** A second port for the failing rows could never be read: when the check fails the
run stops, and when it passes the port is empty. So the failing rows go in the message. A flow that
wants them as data uses filter.

**A failed check is a `DataError`.** The reader's next move is to look at the data, not the flow file
or the run.

**An empty frame fails by default.** Every row of an empty frame meets every condition, which is the
silence a filter that matched nothing leaves behind, the most common failure these checks exist to
catch. `allow_empty` says in writing that a branch may be empty.

**A null result always fails.** filter has a `nulls` setting because both choices are valid when
selecting rows. For a rule that must hold, "could not tell" is never a pass, and a setting to allow it
would exist only to silence the check.

## cast_columns

**A type is stated by someone who knows.** A CSV has no types; an old database column holds digits as
text; two branches about to be stacked disagree because one only saw whole numbers. The fix is the
same each time: say what the column is.

**A step, not a setting on other tools.** It could have been a field on union, read_table and
read_excel, written four times and drifting apart. As a step it serves all of them, and a type change
becomes a box on the canvas with a `why` beside it.

**A value that will not convert stops the run by default.** Making it null by default gives a frame
that looks right and is not: the column has the right type, the row is there, and the value is gone.
The refusal is a `DataError` (the values stopped it, not the configuration) naming the column, the
type and the rows. Owner's ruling (2026-11-18, finding E-01): a decimal becomes a whole number only
when it already is one, because polars would cut 1.99 to 1 silently.

**`set_null` reports and marks.** It suits a large messy source, where three bad values in 46,000
should not stop a nightly build. It counts the failures out of the total and their share, which makes
the step a warning, so the fact travels into the log and onto the canvas. The count is taken before
the values are replaced, because afterwards a failed value and an original null look the same. Each
emptied row also gets a status mark naming the column and quoting the value: a count alone cannot tell
a lost figure from a name that was never a figure. A blank gets no mark. The value is spelled
`set_null` because an unquoted `null` in YAML is no value at all.

**A cast to the type already there is refused.** It does nothing, and the usual cause is a
configuration edited halfway.

**Text is not cast to true/false.** Sources spell it many ways (true, T, Y, 1); case_when decides that
in the open.

**A zone-aware timestamp needs `in_zone` to become a Date** (owner's ruling 2026-11-18, finding E-02).
The day depends on the zone, so there is no default; the migration wrote `UTC` into the client steps
that needed it.

## date_diff

**The count is usually a divisor.** An amount billed for a period is divided by the number of units
in it. A count one too many understates every unit and one too few overstates it, and neither shows:
the result still looks like a figure. So nothing about the count is left to a default.

**`counting` is required** (stage 4 rename, finding A-05/A-23). `inclusive` counts the unit each end
falls in and is never zero; `elapsed` counts completed units only and is one less in size in both
directions, like SQL's DATEDIFF and Excel's subtraction. A default would choose every divisor silently.

**A month counts when the end's day reaches the start's.** 31 January to 28 February is one month;
1 January to 31 December, inclusive, is twelve.

**A backwards range counts negative.** Which end was mistyped is a judgement for a repair step, not for
this tool. A missing date gives null, never zero, for every unit and counting.

**A zone-aware timestamp needs `in_zone`** (owner's ruling E-02). The day depends on the zone.

**One counting function.** date_diff and the expression functions `months_between` and `days_between`
call the same `count_between`, so a tool and a formula cannot disagree.

## assemble_date

**Why it exists.** A date often arrives as parts an earlier step pulled apart: a year, a month, and
often no day, because a period that names a month never stated one. Everything that compares, sorts or
counts dates downstream needs a real date built from the parts the source named.

**The day defaults to 1**, the only day every month has, so the default never depends on the month's
length. The month's end is written down instead: `day: '31'` with `on_invalid: clamp` resolves per row
to that month's last day.

**`on_invalid` covers only a day the month is too short for.** A month outside 1-12, a year no date can
hold, and text that is not digits are refused under every setting, because none of them is a short
month. A null part gives a null date under every setting, `error` included: missing data and
impossible data deserve different answers, and failing a run over rows nobody filled in helps no one.

**Months are numbers.** A source's spellings of month names are a policy for a rule list a reviewer can
see, not something to guess.

**Digits are checked, not cast.** polars' lenient cast reads `'+5'` as 5.

**One build function.** assemble_date and the expression function `make_date` share `date_or_null`.
assemble_date stays a tool beside `make_date` (finding E-19).

## expand_date_range

**Both ends are included, per period.** January to December is twelve rows. The period holding the
start counts however far into it the start falls, and so does the period holding the end, because an
end written as a month's last day is by far the commonest spelling, and dropping that month would lose
a whole period's money from every row. The same day at both ends is one row.

**A backwards range goes to `unexpanded`, judged on the dates.** An end before its start is real data
(a mistyped year, a swapped day and month), and the fix belongs in a repair step where it can be
reviewed. Two ends in the same month the wrong way round are backwards too: month arithmetic alone
would give them one plausible period, the exact shape a swapped day and month makes. This agrees with
date_diff's negative count on the same row.

**A row that cannot be expanded never vanishes and never gets a null period.** The obvious
implementation explodes an empty list into one row with a null period. That is worse than dropping
it: it looks like a schedule row, carries the row's whole figure, and sits in a report as a period that
does not exist. `unexpanded` has exactly the input's columns, with no reason column, so it stays
union-compatible; the reason is one condition away downstream.

**Only `month`.** Every other unit brings its own question (what is a partial week or year worth?) that
nothing has asked yet, and an allowed value with no tests behind it is worse than a short list.

**A zone-aware timestamp needs `in_zone`** (owner's ruling E-02). Timestamps become calendar dates
first, as in date_diff.

## normalize_text

**It exists to make a join key.** The join is only as good as the key, so the key must come out the
same whichever side of a join, and whenever, the step runs.

**`operations` chooses; `OPERATION_ORDER` orders.** The clean-ups enable each other: folding accents
lets `SARL` be recognised as the suffix `sarl`, and removing punctuation lets `Ltd.` be recognised as
`ltd`. A caller-chosen order could run an enabling step after the one it enables, and the step would
stop being idempotent: running it twice would give a different key from running it once.
Whitespace collapsing runs last because removing a suffix leaves the space before it. Listing an
operation twice is refused: it shows the list was read as a sequence.

**Suffix words are configuration.** A jurisdiction's company suffixes are not this library's to know.
Stripping repeats until a pass changes nothing, which also keeps it idempotent; a cap of 64 passes is a
safety margin only.

**Accents use NFKD.** It splits an accented letter into letter plus mark, and the mark is removed. As a
compatibility decomposition it also turns full-width and ligature spellings into plain letters, with
nothing transliterated.

**Punctuation is deleted, not spaced.** `A.B.C.` meets `ABC`, and a curly apostrophe meets a straight
one. Whitespace means any Unicode space, because a name pasted from a spreadsheet often carries a
non-breaking space.

**A key that cleans to nothing is null.** A column of empty strings would join every unusable name to
every other one.

## regex_extract_groups

**Why a tool, when `regex_extract()` exists.** One group is `regex_extract()` in an expression. A
pattern with several named groups gives several columns from one pass and one decision per row: a row
that does not match is null in every group, never in some. A half-matched pattern does not leave half
its groups filled. A group the pattern makes optional can still be null on a match.

**Column names come from the pattern.** Two places to write the same names are two places for them to
disagree. The names are read from polars, which runs the pattern, not from Python's `re`, because the
two disagree on some patterns (look-behind is the case that decides it).

**Groups are text.** Guessing a type would make a column's type depend on which rows arrived.

**`position_column`** exists for flows that try several patterns and keep whichever matched first in
the text: "first stated" is usually what a reader means, since a line names the term it bills before
any other. The offset counts bytes (polars' `str.find`), so on non-ASCII text it is larger than a
character count. Comparing two offsets is always right; slicing text with one is not.

**`all_matches` uses Python per distinct text.** polars can list every match but not every match's
groups. Cutting each match out and reading it alone gives wrong groups when the pattern looks at the
text beside the match, as `\b` and `\B` do. So each distinct text is searched once with
`re.finditer`, a known exception to whole-column work. polars' list of whole matches checks Python's,
and a text where they differ is refused; a pattern that can match empty text is the usual cause.
`match_number_column` is required, because without it two rows from one text cannot be told apart.
`position_column` is refused, because one offset cannot describe several matches.

## match_one_to_one

**Why a tool.** Whether a candidate is still open depends on which candidates were accepted before
it. A join and a filter cannot say that.

**The walk order is total.** `priority` columns first, each in its own direction like sort, missing
values last; then the left key and the right key, ascending. No two candidates are ever in an undecided
order, so the result does not depend on arrival order. Both ports come out in walk order.

**Two ports, like filter.** Every input row leaves on exactly one of `accepted` or `rejected`, with all
its columns.

**A missing key pairs with nothing,** as join follows SQL's `=`. Such a row is rejected and takes no
key.

**`ties: refuse`.** The walk stops at each candidate whose keys are both free and looks for a rival:
another candidate sharing a key, equal on every priority column, and still available (neither key
taken or refused). Two nulls count as equal here, and the key tie-break never counts as a difference. A
key with a rival is refused: it stays unmatched and every candidate naming it is rejected. A refusal
never takes the key on the other side.

**The walk runs on whole columns, in rounds.** Each round decides at once every open candidate whose
outcome no earlier open candidate can change. Under `first`, that is a candidate that comes first for
both its keys. Under `refuse` it must also come before every open candidate that could change whether
its rivals are still available (everything one key away, and one key further). Two candidates decided
in the same round are never that close, so deciding them together gives the same answer as walking one
at a time. The best open candidate always qualifies, so each round decides at least one, and the number
of rounds grows with the longest chain of second choices, not with the row count.

**A polars trap.** Taking one window over another inside a single expression gave wrong answers, so the
first window ("near") is written to a column before the second ("reach") reads it. A regression test
pins that case.

## make_groups

**Why a tool.** Two records are in one group when any chain of pairs joins them. A fixed number of
joins cannot follow that: two joins follow a chain two links deep and lose the third. (Renamed from
`connected_components` in stage 4.)

**Named after the smallest record.** That name does not depend on the order of the pairs or on which
side of a pair a record sits, so the same pairs always give the same frame. Records compare by value
in their own type: 9 before 10, but text 'R-100' before 'R-20'.

**A null endpoint links nothing,** as join follows SQL's `=`. The other end of the row is still a
record. A pair of a record with itself is how a record with no links is listed.

**Types must match.** The number 2 and the text '2' are not the same record, so the tool refuses rather
than guess.
Two exact columns of different places are the exception, as in `union`: 1.50 and 1.5000 are one
record, widened to the larger places. A value too wide to widen is refused by name, never rounded.

**How it runs on whole columns.** Each distinct record gets a number: its position when sorted, so a
smaller number is a smaller record. In each round, every group touching a smaller group points at the
smallest such neighbour; then every record follows its pointers to the end by repeated doubling; then
the links are rewritten between the groups that remain. It always ends: a record only points at a
smaller number, so pointers never loop, and each round the larger end of every remaining link stops
being a group. On a single chain at least half the groups go each round, so 200,000 records in a chain
take at most 18 rounds.

## cross_tab

**For series.** One row per key and period becomes one row per key with a column per period, which is
how a person reads a figure over time.

**Columns are sorted, as text.** A reader scans them across a page, so the order comes from the values,
not from arrival. The key column must already be text: turning a number or date into a heading is a
formatting choice with several right answers, and choosing one here would freeze it into every cross
tab. The formatting step runs in front.

**An empty cell is null, never zero.** "No rows for this key in this period" and "nothing was measured"
are the same fact; a zero is a figure someone measured, and once a made-up zero is in the frame nothing
later can tell it from a real one. polars counts no rows as 0 and sums no rows to 0, so each cell is
guarded. `sum` asks the stricter question "did a value land here", because a cell of only nulls also
sums to 0; group_by rules the same way. The cost: the column names come from data, so no later step can
name them to repair them. The honest fix would be one generic fill step, written once; until someone
asks, the gap stays open rather than being papered over here.

**A null or blank key stops the run.** Dropping those rows would delete their figures from the grid
silently. Giving them a column is worse: any name chosen ("unknown") is a name the data could also
carry, and would merge with it silently. Null and blank are one rule because different extracts write
the same missing period both ways. The fix belongs upstream, where an author decides to drop or label.

**A key value equal to a 'by' column's name is refused.** The name comes from the rows, so it can only
be checked once they are seen. Left alone one column would replace the other.

**The aggregates are group_by's set, imported.** Two lists drift, and a reader who learned one would
write a configuration the other refuses. Each aggregate has its own branch and an unknown one raises:
ending on `max` is how `first` once filled cells with maxima.

**`join_text`** works as in group_by and uses its default separator; one aggregate fills every cell, so
a separator setting would be one more field on a tool whose few fields are the point. `quantile` is
the exception that earns one: without `fraction` it has no answer at all, so `fraction` is a fifth
field, read by `quantile` only and refused beside anything else. A cell with rows
but no values is empty text; a cell with no rows stays null, a distinction group_by never has to make.

**`values_from` is always required**, even for `count`, because one aggregate fills every cell.

## unpivot

**Why it exists.** cross_tab turns long into wide, and there was no way back. A frame with one column
per period cannot be filtered, joined or grouped by period, because the period is a heading, not a
value. One step that turns the headings into a column puts them where every other tool can read them.

**No setting lists the columns to turn.** Every column is in `by` or turned; there is no third kind, so
such a list would say nothing new and could contradict `by`. Its obvious name would also be `on`, which
YAML reads as the boolean true (a trap that has cost this project twice). A setting that does not exist
cannot be named `on`.

**`by` is required and not empty.** Without an identifier, nothing says which row a value came from or
lets it be joined back. A flow that wants that adds record_id first. A `by` naming every column is
refused: nothing would be turned, and a wide frame would go on to steps written for a long one.

**`names_to` and `values_to` have no default.** They are the point of the step; a default would put a
name nobody chose into every later step and report.

**The turned columns must share a type, except an all-null column.** This is union's rule, because an
unpivot stacks columns the way union stacks frames. Text under a number turns money into text silently,
so it is refused, naming both columns and types.

**`values_type` turns a mixed block on purpose.** The case: a block that gains a column each period,
with unrelated text beside it. The text moves as the block grows, so it cannot be named in `by`, and
telling it from a period means reading its heading, which needs the unpivot to have happened. Stating
the type keeps figures as figures; turned as text, a figure a hair under half a cent can come back a
hair over and round the other way. Reading a column as the type it already has is accepted here and
gives no note, unlike cast_columns: there the columns were named one by one, so one already right is a
leftover line; here the set is everything not named, and "some already right" is normal. A decimal with
a fraction does not convert to a whole number (owner's ruling E-01).

**`on_invalid` follows cast_columns.** `error` (when absent) stops the run; `set_null` empties, counts
before replacing, notes only columns that lost a value (most are clean), and marks each emptied row.
The marks follow the output rows, so they are built one block per turned column and stacked like the
values. `on_invalid` has no declared default, because the rule refusing it without `values_type` reads
whether it was written. Known gap, shared with cast_columns: a type pair polars refuses for a whole
column (Date to Float64) raises an ExecutionError, not a DataError.

**Rows come out column by column.** All rows of the first turned column, then the second, in the frame's
column order. Not interleaved (what a naive reshape gives) and not sorted (by something nobody wrote).
The 'by' columns come in the order configured, because the configuration is the author's statement of
the output's shape.

**A null cell is a row by default.** The row existed and the cell was empty; both are data. Dropping it
would make a later count short without looking short. `exclude_nulls: true` follows SQL's UNPIVOT
default (EXCLUDE NULLS). Only null is left out; empty text, spaces, zero and not-a-number are values. A
cell is judged as it arrived, so text that failed to convert stays, marked. The option exists to save
memory on sparse sheets, so null cells are dropped per column before rows are built; filtering after
would give the same rows and save nothing. A test measures peak memory.

**Every refusal is read from the schema,** so an empty frame is refused the same way. A refusal that
needs a row would report success the day a filter upstream matched nothing.

## read_csv

**A sibling of read_excel, not a format switch on it.** A cell is typed and a text file is not, so
this tool's first job is working out types, and its hazards (a ragged line, an undeclared encoding, a
blank that is not empty text) are not a workbook's.

**Types are worked out from the whole file.** There is no `infer_schema_length`: a sample would make a
column's type depend on where this month's odd value sits. `column_types` corrects only the columns
inference gets wrong. Dates stay text unless declared, because a file cannot say which order its digits
are in.

**A bad value or a ragged line stops the run.** Padding a short line, or turning a bad value into null,
both give a frame that looks right and is not. The refusal names the line or the value.

**A number that would change a value is refused.** A leading zero (02134, a postcode) is lost by a
number, and a whole number past 64 bits can only be read as a float, which rounds it. The author
decides in `column_types`.

**A blank field is null, and a blank line is a row.** write_csv writes a row of nulls as an empty line,
so dropping it here would lose that row on a round trip.

**The encoding is a fact about the file,** from a closed list so a typo is refused at load. A byte
order mark is dropped; invalid bytes are refused, never read lossily.

**Parsing.** A file with no quote character is split directly; otherwise Python's csv reader handles
quoted fields that contain the delimiter or a line break.

## read_excel

**A sheet is named, never an index.** "The first tab" is a guess dressed as a default: a delivered pack
often has a cover sheet in front of the figures, for exactly the readers who never configure a flow. A
name that is not there is refused with the names that are, in the words used for a missing column. A
near miss (other case, trailing space) is not a match. This mirrors read_table, which accepts a table
name and never a query.

**What comes back is what the sheet holds.** Whole numbers stored as floats come back whole. fastexcel
decides a column's type from its first 100 rows, so text in those rows makes the column text. Text
further down a column of figures, or a word fastexcel treats as missing (`n/a`, `#N/A`), would read as
null; the read stops instead and points to `as_text`. The check reads the sheet a second time as text
and compares cell by cell.

**The header is the schema.** A repeated, blank, date or true/false heading is refused, not repaired,
and two headings that read the same (2026 and "2026") count as repeated. Repaired, fastexcel's own
names (`key_1`, `__UNNAMED__1`) would leak into column names the moment a later step names one.

**No cell range.** `skip_rows` reaches a header below a title block; narrowing columns or rows is
select_columns or filter after this step.

**`.xls` is read by the same rules, and that is measured.** An .xls file stores dates as days from 1900
or 1904 (its header says which), text in a table of its own, and errors and merged cells in records of
their own. fastexcel reads both formats through calamine, and the test builds an .xlsx from the same
cells as two committed .xls fixtures (one per date system, made with xlwt; the script beside them says
how) and requires the same frame or the same refusal for every sheet, typed and `as_text`. They agree on
every case measured. Two cases agree but are worth knowing: an error value (`#N/A` from a formula) reads
as null in both formats and is not refused, and a time-only cell in a 1904 workbook reads on 1904-01-01
with `as_text`. Refusing error values would stop flows that read such cells today, so it waits for a
release that says so. calamine picks its reader by the file's ending, so a web page or text file named
.xls (a common export) fails inside the engine; the first bytes are checked to say what the file holds.
`.xlsm`, `.xlsb` and `.ods` stay refused until someone measures them the same way.

**`sheets_matching` stacks by name.** Not one read step per tab, which has to be edited each time the
workbook gains a tab. Each sheet is read as if named alone. Stacking lines columns up by name; whole and
decimal become decimal; an empty column takes the others' type; any other difference is refused.
`sheet_column` is required, since rows from different sheets must be told apart.

**`as_text` reads four times.** fastexcel reads a whole column as one type. To give each cell the text
of what it stores, the sheet is read as text, figures, true/false and dates, and each cell takes the
reading that suits it. The text reading alone is faster but rounds figures (99999999999.99 comes back
99999999999.990005493). Excel stores dates and times as one number, so midnight reads as a date and a
time-only cell reads as that time on 1899-12-31. A test pins each case.

**Lazy import.** fastexcel and xlsxwriter (for the examples) are imported inside the functions that use
them, so importing the package does not load them (findings G-17 / B-21). A missing fastexcel is refused
with the name of the package to install. A test imports the package in a fresh interpreter and checks.

## write_csv

**A sibling of write_excel, not a format switch.** A text file has no row limit, no cell holding a
double, and no tab-name rules, so none of the workbook's refusals apply; and text has hazards of its
own. Whole numbers and Decimals are written digit for digit.

**Quoting follows the format's own convention.** Only a value holding the delimiter, a quote or a line
break is quoted, wrapped in `"` with inner quotes doubled. read_csv assumes this, so it is not a setting.

**UTF-8 without a byte order mark, line feeds.** Not settings: a writer chooses, while a reader
discovers a fact about someone else's file. Hence read_csv has `encoding` and this tool does not.

**Null and empty text are the same on the page.** polars would quote an empty string to keep them apart;
that is a distinction this format is not entitled to keep, so empty text is written as null. read_csv
reads both back as null. The loss is one-way and stated.

**An empty frame stops the run unless `allow_empty`,** as on the table writers: an empty file from an
upstream failure looks like a quiet month. Allowed, a header-only file is written, so the delivery is
visibly empty rather than missing.

**List, Struct and Duration are refused,** each having a plausible wrong spelling as text. Every check
runs before the file is opened.

**`overwrite`.** A file at `path` is refused at load unless `overwrite: true`. A flow that runs nightly
needs it from its first night, or its second run will not load. Saying true when nothing is there is
accepted.

## read_table

**A schema and a table, never a query.** A `query` setting would be arbitrary SQL in a configuration
file: its column references could not be checked before the run, its dialect would tie a portable flow
to one database, and it would repeat filter and select_columns in a language none of the library's
checks can read. So a step names two identifiers, each checked against a plain-name grammar before a
connection opens (the only defence, since a name cannot be a bound parameter). Narrowing is a filter or
select_columns after it, where the columns are visible to the same checks as every other step.

**`connection` names an environment variable.** A pipeline file is then safe to commit and to hand to
another client as a template. `db.py` holds the rest of that ruling.

**Types come from every row.** A column null for its first hundred rows and filled later is ordinary
(an id only recent rows carry, a field added by a later migration). Sampling would infer Null and then
fail on the first value, or infer too narrow a type silently. Reading every row costs one pass and
means the tool cannot invent a schema the table does not have.

**An empty table is typed from its definition**, where that is certain; a column whose values could be
of more than one kind (numeric with no places stated, zone-aware timestamp, most SQLite types) stays
Null. A read with rows never takes this path.

**A Postgres numeric is read as its text and made exact by the tool.** The query names each numeric
column as `"col"::text`, and the tool converts the text with the same plain-digit reader every exact
conversion uses. That gives one answer whichever driver is installed, types a `numeric(p,s)` from its
declaration even with no rows, and turns a value no exact column can hold (more than 38 digits, NaN,
Infinity) into a refusal naming the column, where the engine would raise its own error. A `numeric`
with no places stated keeps arriving at the most places in its data, as it did when the driver's
Python decimals decided; the notes recommend declaring the places at the source, because otherwise a
night with one 3-place value changes the column's type. An exact column from any driver is widened to
38 digits in all, so `numeric(12,2)` and a computed 2-place column always match. SQLite has no exact
type and is read as before.

**Rows come in no particular order.** A database promises none without ORDER BY; a step that needs
order sorts first.

## write_table

**`replace` is one transaction.** It empties the table and loads the frame as one unit, so a write that
fails part way (a NOT NULL column fed a null, two rows claiming one key) leaves yesterday's figures,
not an empty table. The failure it rules out: a build that deletes yesterday's numbers, falls over on
row 400, and reports an error over an empty table.

**It never creates a table.** A table made from what a frame inferred today has types decided by one
day's data and no key, no NOT NULL, no index (the constraints that make replace safe), and a typo in
`table` would quietly create a second table and leave the real one stale. DDL belongs in a reviewed
migration.

**`mode` has no default.** `replace` and `append` differ in whether running a flow twice is safe.

**Columns must match exactly, in any order.** An extra column would be dropped, a missing one filled
with null where a figure belongs; a case-only difference is refused, not guessed. Values that will not
fit a column's declared type (text in an integer column, a fraction) are refused before anything is
written. These checks live in `_table_load.py`, shared with publish_tables.

**An empty frame stops the run unless `allow_empty`.** A replace of nothing is the loudest and most
expensive silence a nightly build can leave.

**The output passes the rows on,** not a receipt: a receipt loses the data a flow may want to write
twice or check, and its interesting fields would drift between runs in a way an example cannot pin.

**Finding the offending rows.** When the database refuses a load, the step names the rows. Two routes
must give the same answer. Replaying the load row by row costs three round trips per row (for 130,000
rows, a failed build that looks wedged), so the table's rules are read from the catalog first and the
frame is judged in memory: missing required values per column, then repeated keys walked in order,
keeping only keys from rows actually accepted (a refused row's key stays free; judging the frame as
written blames the wrong rows once two key rules are in play). When appending, keys already in the
table count, fetched in one statement. This route declines, falling back to the replay, when the table
has a rule it cannot judge (a check expression, a foreign key), when SQLite would fill a null rowid key
itself, when reading the rules fails (logged, since it would otherwise show only as slowness), or when
it finds nothing wrong with a load the server just refused. The replay attempts every row inside a
discarded transaction, each in the loader's own undo, because on one driver a failed statement poisons
the rest of the transaction.

## publish_tables

**All the tables, or none.** write_table is one transaction per step. Tables that only make sense
together (rows, a series, an audit, a change log, a dimension) written one step each are that many
transactions, so a build that fails half way leaves three tables from tonight and two from last night,
published, disagreeing, and with no record of which is which. Nothing passes a transaction between
steps, so it has to live inside one step. Every ruling below follows from that.

**Input ports come from `tables`,** as in write_excel (`input_ports_for`, owner's approval for tools
that declare their own ports).

**One instant, read once.** The stamp belongs to the publish, not to any table or frame. Read per table,
tables of one build would carry times milliseconds apart and could no longer be selected as one build;
taken from a constant or a frame, every build would look the same. UTC without a zone, because both
engines store a plain timestamp and a set of figures read a year later elsewhere must still say when it
was built. Renamed `stamp_column` in stage 4.

**A merge states what a null does** (owner's ruling 2026-11-18, finding D-02). `when_incoming_is_null`
has no default. `keep_published` writes `coalesce(excluded.c, target.c)`: one unreadable upstream source
cannot blank a column across every row. `overwrite` writes `excluded.c`, as SQL's MERGE ... UPDATE SET
does. The client's existing entry was migrated to `keep_published`, so nothing it publishes changed.

**A merge stages first.** One driver's bulk path is COPY, which has no conflict handling, so rows are
bulk-loaded into a temporary `_etl_merge_staging` table and upserted in one statement. One fixed name,
created and dropped inside the publish; a permanent table of that name is hidden while a merge runs.
A merge needs one row per key: a null key would add a row on every run, and a repeated key would blend
two rows into one.

**Append never truncates, and `mode` has no default.** A change log is the one table a rebuild cannot
recreate. The three modes differ in whether running twice is safe.

**It never creates a table,** for the same reasons as write_table. Values that cannot go into a
column's declared type are refused before anything is emptied (SQLite stores 'heavy' in an integer
column and reports success). The checks are shared with write_table in `_table_load.py`.

**Columns are compared both ways.** A column the table has and the frame lacks is refused, as is the
reverse. A column added to a table by hand and not to the flow once stopped every publish for twelve
days; the refusal was correct, and nothing reported it. A consumer needs its own freshness alert.

**Every refusal comes before `execute`.** A refusal after four tables have been emptied is not a
refusal; it is a failed publish with a helpful message. Repeated ports and tables are refused because
both fail silently: one port publishes the same rows twice, and two loads into one table in one
transaction leave only the second.

**The receipt is one row,** the time and a row count per table. One pass-through port per table would be
ports nobody wires; a count per table is what someone comparing tonight with last night reads.
