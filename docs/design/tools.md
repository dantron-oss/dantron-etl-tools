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

## group_by

**Aggregates are settings, not expression functions.** The expression language works one row at a
time and does not list `sum`. Allowing `sum([amount])` in one setting only would make a second
dialect a reader has to learn from where a string sits. So group_by has its own short list,
`AGGREGATE_FUNCTIONS`, and the `function` setting's allowed values are built from it, so the two
cannot drift. cross_tab imports the same list, so adding an aggregate adds it to both.

**No `mean`.** No flow has wanted it, and its denominator depends on how nulls are counted, which is
the kind of question this tool refuses to answer quietly.

**Groups keep first-appearance order.** A key of several columns has no obvious sort order, and the
order the input already has is the one a reader knows. polars promises no group order unless asked,
so `maintain_order=True` is set.

**A null key is a group.** This follows SQL's `GROUP BY` and agrees with remove_duplicates. Dropping
those rows would delete their amounts from every later total with nothing left to say they were
there. join decides the other way because it follows `=`.

**A sum of nothing is null.** polars sums an all-null group to 0. Passed on, that is a figure nobody
measured, and nothing later can tell it from a real zero. Nulls are still skipped in a group with
any value, so all-null is the only case that gives null.

**`first` and `join_text` read arrival order.** `first` exists for a column that is the same on every
row of a group (a document number on its lines), where any row's value is all of them. When the
column does vary, `first` answers with whichever row came first, a fact about order, not data. A
flow that needs a repeatable answer sorts first, and can add `count_distinct` beside it to check the
column really is constant. Owner's ruling (2026-11-18): `first` keeps the first row's value even when
it is null; `first_non_null` was added for the other reading.

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
