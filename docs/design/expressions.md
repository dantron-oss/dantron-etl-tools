# The expression language: design notes

Why the bracket expression language (`[amount] > 0`, `round([total], 2)`) works as it does. The code
in `src/dantron_etl/expr/` says what it does; these notes keep the decisions, the alternatives that
lost, and what was measured.

## Why never eval

A tool's configuration is a file, and a person who is not writing code fills it in. A filter needs a
condition and a formula needs an expression, so this package turns text such as
`[status] = 'PAID' and [amount] > 0` into a polars expression. Every character of that text comes
from outside the program, so this package is an execution boundary.

Nothing the author writes is ever executed. There is no `eval`, `exec` or `compile`, no attribute
lookup driven by the text, and no hand-off to a database's own expression parser. The steps are:

1. **Scan** (`scan.py`). A forward pass rewrites the author's syntax: `[name]` and every quoted value
   are lifted into tables and replaced by numbered calls (`col(0)`, `lit(1)`), and `=` becomes `==`.
   No character the author typed reaches the parser. The internal names are refused in the incoming
   text, so the only way to produce one is to write `[name]` or `'text'`.
2. **Parse** (`parse.py`). `ast.parse(..., mode="eval")` builds a tree. Nothing runs.
3. **Check** (`parse.py`). Every node is checked against `ALLOWED_NODES`, and anything else is refused
   by name. It is a whitelist, so the question a reader asks is "why is this node allowed?", never "did
   anyone think of this one?". Attribute access, subscripts, comprehensions, lambdas, binding,
   unpacking and string formatting are all absent.
4. **Compile** (`compiler.py`). The checked tree is translated into a polars expression, with calls
   limited to the function table.

A scan, not a regular expression, does the rewriting, because `[note] = 'a=b'` and `[note] = 'x [y]'`
contain characters the rewrite touches; a blind substitution corrupts the literal, and the corruption
then looks like a data problem.

**The language grows by functions, never by syntax.** No user-defined functions, no attribute access,
no subscripts, no lambdas. That refusal is a security boundary: a pipeline file is edited by people who
are not developers. One more named function needs no new grammar to document, review and teach.

## Limits come before parsing

Building the tree is itself the fragile step: a long enough operator chain exhausts the parser's
stack. So the length limit (`MAX_TEXT_LENGTH`) is checked first, the depth limit (`MAX_DEPTH`) is
checked with an explicit stack, not recursion (a recursive check would crash on the input it exists
to refuse), and `RecursionError` and `MemoryError` from the parser become ordinary refusals. A flat
list of ten thousand values passes the depth limit, so a breadth limit (`MAX_NODES`) refuses it by the
limit that actually applies. `MAX_NODES` sits below half of `MAX_TEXT_LENGTH`, since about two
characters buy one node.

The limits are public because they are part of the contract: a limit an author cannot see is one they
cannot design around. They are generous next to any real condition (the widest is a membership test
against a list of codes); they bound work on untrusted text and do not shape the language.

## The function table

Every function is defined once, in `DEFINITIONS` in `functions.py`: its arguments in order and what
each must hold, the family and exact type of its result, the builder, a one-line summary and an
example. `ALLOWED_FUNCTIONS`, the arity check and the result types are all read from the entry, and a
test refuses an entry with a part missing. The manual's list (section 4.2) is checked against it.

- **No default arguments, with one exception.** One spelling per call keeps a file reviewable; a
  reviewer should not need to know a default to know what a line does. `substr()`'s length may be
  left out (an entry's `optional`), because "to the end" is the only reading, and the one SQL gives.
- **The result family is declared,** so nesting is checked: without it `upper(len([label]))` would
  compile and fail in polars.
- **The exact result type is declared where known** (`year()` and `len()` give whole numbers), so a
  count of months built from them is known to be whole.
- **Literal-only arguments.** A regex pattern, a group number, `round()`'s places, `format_number()`'s
  places and separators, `regex_replace()`'s replacement and `uuid5()`'s namespace must be written in
  the file. So must the counts of `left()`, `right()`, `substr()`, `split_part()`, `pad_left()` and
  `pad_right()`, the delimiter, the fill character and `parse_date()`'s format: like `round()`'s
  places, they say how text is laid out. A decision a reviewer reads in the line cannot vary by row,
  and a literal can be checked before the run. `add_months()` and `add_days()` are the other way:
  their count is a quantity, and comes from the data.
- **A row with no answer stops the step.** An entry's `no_answer` names the rows a function cannot
  answer (text `parse_date()` cannot read; a count `add_days()` cannot move by exactly), and
  `divides_by` names `mod()`'s divisor. Each tool already checks these with the division check, so
  such a row stops the step with the values quoted, instead of becoming a null nobody asked for.
- **A pattern is checked by polars, both ways,** because polars and Python's `re` disagree (look-behind
  is legal in `re` and not in polars; a Unicode script class is the reverse). Checking with the wrong
  engine either passes a pattern that fails at run time or refuses one that works.
- **`contains()` is literal.** A substring test that accepted patterns would make a dot in an account
  code mean "any character". `regex_match()` is the way to ask a pattern question and get true/false.
- **`regex_extract()` refuses a group the pattern does not have,** counted by polars with no row read,
  so a pattern without brackets asked for group 1 is caught in the file, not left null on every row.
- **`regex_replace()` replaces with text, verbatim.** polars reads `$name` in a replacement as a group
  and deletes it when there is none, so `'$5.00'` would become `'.00'`. The `$` is doubled, and the
  pattern is wrapped in `(?:...)` so polars always takes the path where doubling means one thing (a
  one-character pattern otherwise takes a fast path that leaves the doubled sign).
- **`concat()` gives null if any piece is null.** A key built from parts with a missing part dropped is
  a shorter key that matches the wrong rows; a null key matches nothing and shows in a count. To get the
  other behaviour, write `concat(coalesce([code], ''), '-x')`.
- **`len()` counts characters,** not rows: an expression is one column's values.
- **`add_months()` clamps:** 31 January plus one month is 28 February, not 3 March. polars does this in
  both directions. The count may come from a column, so it is built into polars' offset text (`'3mo'`).
  A count that cannot be shown to be whole is refused (see "Types and widths").
- **`months_between()`, `days_between()`, `make_date()`, `month_end()`** (added in stage 5 because flows
  were casting dates to integers to subtract them) share `dates.py` with date_diff and assemble_date:
  `count_between`, `date_or_null` and `days_in_month`. A tool and a function cannot disagree.
- **`uuid5()` is the standard library's,** run row by row. The point is that the derivation is not this
  library's: an id only this library can reproduce is not a key. A hand-written SHA-1 measured 2.4
  times faster and was refused; the contract is equality with `uuid.uuid5`. polars has no SHA-1 (its
  `hash()` is a different, non-cryptographic one), so there is no column kernel. A batch-at-a-time map
  measured 6% faster and would need null handling written by hand; not worth it. About 5 microseconds
  a row. A null gives a null id: hashing a stand-in would put every row missing a part under one
  real-looking id. The namespace is read from the file and checked with the same `uuid` library before
  any row.

## Rounding like a spreadsheet: 15 significant digits

Owner's ruling (2026-11-18, finding C-03): `round()` takes halves away from zero, like Excel, on the
decimal as written, so `round(1.005, 2)` is 1.01.

1.005 is held in binary as 1.00499999999999989..., below the half, yet it is written 1.005 and a
spreadsheet rounds it to 1.01. So a Float64 is read to 15 significant digits (a Float32 to 6): fifteen
is the most a 64-bit double always carries faithfully from decimal and back, and what a spreadsheet
works to. A value that differs from a half only past its fifteenth digit is treated as a half. In
practice the value is moved away from zero by half a unit of its fifteenth digit, then rounded half
away from zero; the move is far too small to change any value not within noise of a half. When the
digit being rounded on is past the fifteenth, the value comes back unchanged. Whole numbers are already
whole; an exact Decimal type is rounded on its own digits; null, NaN and infinity pass through. With no
schema, the type is unknown when the expression is built, so the choice is made per batch.

**`format_number()` writes what `round()` gives.** The value is rounded by `round()`'s rule, then
written, so -15.215 at two places is -15.22 from both, and they can never disagree in the last place. A
negative value that rounds to zero keeps its minus sign (-0.004 reads -0.00), the one sign that the
figure was below zero. The digits are grouped in threes with the separators stated, row by row: polars
has no kernel for that, and the cost is acceptable because formatting belongs at the end of a flow, on
the rows that reach a report. Details in `formatting.py`:

- The sign is kept out of the grouping, or `-1832` would group as `-,832`.
- Each value goes through an exact `Decimal`, so a whole number is written with its exact digits and a
  double with the digits that double really is.
- `copy_abs()`, not `abs()`: `abs()` on a Decimal rounds to the process-wide context, so a precision set
  by any library would move a published figure (651,623.80 becomes 651,620.00 under prec = 5).
- The rounding mode is stated explicitly, so another thread's context setting cannot move a figure.
- Infinity and NaN are written as null: a cell must tell "no figure" from a figure (and `Decimal(inf)`
  would group as `In,fin,ity`). NaN is checked before the sign test, which would otherwise raise.
- A null stays null: a formatted 0.00 where nothing was measured looks like a real zero.

## Division by zero

Owner's ruling (2026-11-18, finding C-02): **a division whose divisor is zero stops the run.** Left to
polars it gives infinity, or not-a-number for 0/0, and a later filter reads not-a-number as larger than
every number. The compiler records every `/` with its divisor (`divisions()`), and each tool calls
`refuse_rows_without_an_answer` before running, which names the field, the expression, the divisor and up to
five rows. Zero means 0, 0.0 or -0.0. A null divisor is not zero: the division gives null. A divisor
written in the file as zero is refused at compile time, since it would stop every row. case_when checks
each division only on the rows that reach it, as SQL's CASE does, so an earlier case can guard it.

`/` is true division everywhere: 3 / 2 is 1.5.

## Types and widths

**Families are coarse on purpose:** text, number, date, true/false. Whole numbers and decimals are one
family, because separating them would refuse `[qty] > 2.5`, which polars answers correctly; a check
stricter than the engine takes away capability for nothing. Null belongs to every family. A column
with no schema is `UNKNOWN`, which means unchecked, never refused, so a caller without a frame still
gets every other check. An unrecognised dtype is `UNKNOWN` too, rather than a guess.

**Three levels of checking,** chosen by what the caller passes to `compile_expr`: a schema checks that
columns exist and that each operation suits what they hold; a list of names checks existence only; and
`None` checks neither (the loader validating a definition before any frame exists). A literal's family
is known from the text alone, so `'100' + '20'` and `concat([code], 1)` are refused even without a
frame.

**Mistakes fail here, in the author's words, never later in polars'.** Some checks are stricter than
polars, which would answer silently, because a condition that matches nothing while raising nothing
looks in a log like a quiet day:

- `and`/`or` need true/false operands; on numbers polars would do bitwise arithmetic. Write `[qty] > 0`.
- `+` on text is refused: polars would join two amounts imported as text into `'10020'` instead of 120.
  `concat()` is the name for joining.
- Mixed families in a list, a comparison or `coalesce()` are refused: polars would turn a number into
  text to sit beside text, without a word.
- A timestamp compared with a date reads the date as midnight. That is right for `>=` and `<`, and wrong
  for `=`, `in` (false after midnight), `<=` (leaves out the day: the classic month-end slip), `>` and
  `!=`. Those are refused.
- `in` takes only a bracketed list of literals. A column on the right would be a row-by-row join nobody
  reviewing the file would see. polars types a list from its first value, so a list mixing whole
  numbers and decimals is widened here first.

**Widths.** `len()` and some columns hold unsigned numbers, and `3 - 5` on those wraps to 4,294,967,294,
so unsigned operands are cast to Int64 before arithmetic. A whole-number literal outside Int64, or a
decimal place or group number beyond a few thousand, is refused: polars would reject it with its own
error type, breaking the promise that every bad expression raises one error. The bound is measured on
the value as written, sign included, so the most negative Int64 is spellable.

**Exact types where one check needs them.** A family calls 3 and 1.5 both numbers, but `add_months()`
must tell them apart. So each compiled part carries its dtype where it can be worked out: from the
schema, from the literal, or from the parts (arithmetic and `coalesce()` combine theirs; `abs()` and
`round()` pass on their subject's; division always gives a decimal). A count of months that is a
decimal column, a decimal literal, the result of division, or of unknown type is refused: half a month
is not a distance, and a 1.5 usually means a division someone forgot. A null count gives a null date.
With no schema nothing is refused, as everywhere.

**Compact text columns are cast at the leaf.** Categorical and Enum columns store text compactly, and
polars' string functions only work reliably on plain strings. Casting each reference to String where it
is read means every operation built on it meets a plain string, even through `coalesce()`. The result
is a plain String column, so it will not silently pair with an uncast Categorical column as a join key.

**Null follows polars, not intuition.** A comparison with null is null, and a filter keeps only true
rows. Nothing turns null into false. `is null` and `is not null` are the only tests never null
themselves, which is what makes them the way to find a missing value; `coalesce([code], 'Z') = 'Z'`
lets a null row take part.

## Exact decimals

Owner's rulings (2026-11-20): arithmetic on exact decimals is exact and only `round` rounds; a
number typed in a pipeline is exact as written; an exact value meeting a float stops the step.

**A typed number keeps its text.** The scan passes a number's characters to the parser unchanged,
and the parser reads only ASCII outside quotes, so a number node's place in the parsed text is the
author's spelling of it. `Spelling.written_number` reads it back as a Python `Decimal`: `1.10` keeps
its two places. Each compiled part carries that value as `constant`. Beside an exact column the
constant is used, as a literal with its written places; anywhere else the float is used, as before,
so a flow with no exact column computes what it always did.

**Every answer is checked before it is worked out.** The engine's exact arithmetic is not safe to
hand a value: it stops with its own error when a sum passes 38 digits, gives a blank when a value
with many whole digits meets one with many places (it writes both at the larger places first), and
rounds a product half to even. So each exact operator (`exact.py`) builds two columns: the answer,
and a true/false column that is true where the answer would need more than 38 digits. On those rows
the operands are replaced by null before the engine sees them, so the engine never fails and never
wraps round. The compiler registers the true/false column as a row check, and every tool already
runs row checks (`refuse_rows_without_an_answer`) before it writes anything, so the step stops and
names the rows. In `case_when` a row no case reaches is not checked, and its unused answer is null.

- `+` and `-`: both sides are written at the larger places first; a side that does not fit there
  is too large. Otherwise the sum is too large only when both lean the same way and `|a| > 38 nines
  - |b|`, worked out on values that each fit.
- `*`: the engine writes both sides at the larger places before it multiplies, so it rounds the
  product half to even, and a large value times a small one comes back blank. The library multiplies
  the digits each side holds (812.40 holds 81240) as 128-bit whole numbers, read from the value's
  own text, and puts the point back through text. Those hold 38 digits, so a product of two of them
  is checked first: split each at 10^19; if both high halves are above zero the product is at least
  10^38; otherwise one cross term is left and the product reaches 10^38 exactly when
  `cross + low // 10^19` reaches 10^19. No step of the check passes 128 bits.

## Quoting and literals

- **Text is single-quoted; a quote inside is doubled**, as in SQL and Excel: `'O''Brian'`. The doubling is
  undone in the scan, because Python's parser would read two adjacent literals as one joined string
  (`O` then `Brian`), a plausible value nobody wrote. `quote_text` and `escape_text` are public so a
  caller writing a value into an expression escapes it here rather than by its own string formatting,
  the one route by which author characters could reach the parser.
- **Dates are `date'YYYY-MM-DD'`.** A tagged literal, not a cast such as `'2026-01-01'::date`: one cast
  operator invites every other cast and grows the grammar. Any other prefix Python accepts (`f`, `b`,
  `r` ...) is refused by name.
- **Numbers have one spelling**: digits, a point, an exponent. `0x10` (sixteen, from someone thinking
  of a code), `1_000` (a separator this language lacks) and anything touching the end of a number are
  refused. `1e400` (silently infinity, so `[amount] < 1e400` keeps every row) and `1e-400` (silently
  zero, turning a tolerance into "anything") are refused; `1e-9` stays legal because that is how a
  tolerance is written.
- **`#` and `\` outside quotes are refused**: Python would start a comment and drop the rest of the
  line, or join two lines, both silently.
- **Words from other languages** (`AND`, `<>`, `&&`, `like`, `if(`) get a hint, but only after the parser
  has refused the text, matched as whole words outside quotes and brackets.

## Refusals

There is one error, `ExpressionError`, a `ValueError` subclass so callers that already treat bad input
as `ValueError` keep working. Every message names the problem, says what to write instead where there
is an obvious answer, and quotes the expression, truncated, because it is read in a log hours later by
someone with a file of many expressions. Operator refusals give both the typed characters and the node
type. A part of the expression at fault is spelled back from the scanned tree (only the scan knows
which index is which name), so a refusal about a line with four columns says which one. A check that
cannot see the text raises `Refusal`, and the compiler adds the text.

## The SQL target that was removed

See `sql-target.md`.
