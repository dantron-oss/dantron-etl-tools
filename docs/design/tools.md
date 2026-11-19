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
