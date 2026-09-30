# CrowsNest Rating System

The current rating system uses Plackett Luce from openskill.py in a sequential manner rating one race after the other.

### Features that I think are good

#### clear starting rating

It is confusing when people don't start at the same rating, why do some people suddenly start higher than others?

#### Rating gain / loss

It is fun to see how each race changes your rating number, not just suddenly jump up by 100 points in one race with seemingly no explanation. This is why most video game rating systems actually have two numbers tho, the "hidden" rating, and the visible one which more directly tracks the performance vs the the hidden one for the actual skill.

I'm not sure how that would really work for crowsnest though, where we display a visible rating number. If people are ranked on the hidden rating, then it is confusing why some people are ranked higher even though they have a lower visible number so this solution is not ideal.

#### Objectivity

I like that the current rating system is fairly objective. I don't want any magic numbers, or subjective conference penalties, the rating system should just run and work.

### Downsides of current rating system

#### Lack of conference mixing

The geography of the US makes it hard for the conferences to sail against eachother. This is not too much of an issue for the northeast conferences, but especially the west coast and southern conferences almost never sail against other schools outside of their conference. This causes massive inflation and many sailors from these conferences have some of the highest ratings in the country, despite their skill level not matching that rating at all. They have just never been compared to sailors who have lower ratings but are actually better.

My solution to this problem was to track the number of "comparisons" between each sailor and a sailor from another conference. So a race against 6 sailors from any other conference (PCCSC and NWICSA so all of the west coast count as the same conference here) would get you 6 comparisons. Sailors who have more than 150 comparisons are eligible to be ranked, or count towards their team's ranking.

#### Low level inflation

It is very easy to gain rating at low level events, compared to high level events. A person who has only sailed low level events will almost certainly have a higher rating than one who has sailed the same number of harder events, just because they performed better even though there skill level might be lower.

#### Lack of lower tier team ratings

Schools that aren't even allowed to compete in cross conference regattas won't ever be able to be ranked with the current system since they won't have any comparisons.

### The main issue

My thoughts are as follows. If we consider each sailor as a node in a graph network, there just aren't enough connections to make the current system work effectively. Too many people are isolated and only connected back to the main graph by one or two connections once or twice removed. The current system only shuffles the rating points around within these isolated pools, and has no way of moving large amounts of rating from one conference towards another.
