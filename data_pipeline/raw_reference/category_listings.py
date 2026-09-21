# Real title lists captured directly from books.toscrape.com category listing pages
# (fetched live during this session — title + category + availability are page-rendered
# text, confirmed against the live site). Used only to attach the correct `category`
# label to rows in complete_books.csv (which itself is a verified, byte-for-byte-matching
# public mirror of the same live site's title/price/rating/stock data — see README for
# provenance notes explaining why this session joins two real sources instead of a single
# live requests.get() call).

CATEGORY_LISTINGS = {
    "Mystery": [
        "Sharp Objects", "In a Dark, Dark Wood", "The Past Never Ends", "A Murder in Time",
        "The Murder of Roger Ackroyd (Hercule Poirot #4)", "The Last Mile (Amos Decker #2)",
        "That Darkness (Gardiner and Renner #1)", "Tastes Like Fear (DI Marnie Rome #3)",
        "A Time of Torment (Charlie Parker #14)", "A Study in Scarlet (Sherlock Holmes #1)",
        "Poisonous (Max Revere Novels #3)", "Murder at the 42nd Street Library (Raymond Ambler #1)",
        "Most Wanted", "Hide Away (Eve Duncan #20)", "Boar Island (Anna Pigeon #19)",
        "The Widow", "Playing with Fire", "What Happened on Beale Street (Secrets of the South Mysteries #2)",
        "The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1)",
        "Delivering the Truth (Quaker Midwife Mystery #1)",
    ],
    "Travel": [
        "It's Only the Himalayas", "Full Moon over Noah's Ark: An Odyssey to Mount Ararat and Beyond",
        "See America: A Celebration of Our National Parks & Treasured Sites",
        "Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel",
        "Under the Tuscan Sun", "A Summer In Europe", "The Great Railway Bazaar",
        "A Year in Provence (Provence #1)", "The Road to Little Dribbling: Adventures of an American in Britain",
        "Neither Here nor There: Travels in Europe", "1,000 Places to See Before You Die",
    ],
    "Historical Fiction": [
        "Tipping the Velvet", "Forever and Forever: The Courtship of Henry Longfellow and Fanny Appleton",
        "A Flight of Arrows (The Pathfinders #2)", "The House by the Lake", "Mrs. Houdini",
        "The Marriage of Opposites", "Glory over Everything: Beyond The Kitchen House",
        "Love, Lies and Spies", "A Paris Apartment", "Lilac Girls",
        "The Constant Princess (The Tudor Court #1)", "The Invention of Wings",
        "World Without End (The Pillars of the Earth #2)", "The Passion of Dolssa",
        "Girl With a Pearl Earring", "Voyager (Outlander #3)", "The Red Tent",
        "The Last Painting of Sara de Vos", "The Guernsey Literary and Potato Peel Pie Society",
        "Girl in the Blue Coat",
    ],
    "Classics": [
        "The Secret Garden", "The Metamorphosis", "The Pilgrim's Progress",
        "The Hound of the Baskervilles (Sherlock Holmes #5)", "Little Women (Little Women #1)",
        "Gone with the Wind", "Candide", "Animal Farm", "Wuthering Heights",
        "The Picture of Dorian Gray", "The Complete Stories and Poems", "Beowulf",
        "And Then There Were None", "The Story of Hong Gildong", "The Little Prince",
        "Sense and Sensibility", "Of Mice and Men", "Emma", "Alice in Wonderland",
    ],
    "Fantasy": [
        "Unicorn Tracks", "Saga, Volume 6 (Saga (Collected Editions) #6)",
        "Princess Between Worlds (Wide-Awake Princess #5)", "Masks and Shadows",
        "Crown of Midnight (Throne of Glass #2)", "Avatar: The Last Airbender: Smoke and Shadow, Part 3",
        "A Court of Thorns and Roses", "Throne of Glass (Throne of Glass #1)",
        "The Glittering Court (The Glittering Court #1)", "Hollow City (Miss Peregrine's Peculiar Children #2)",
        "The Star-Touched Queen", "The Hidden Oracle (The Trials of Apollo #1)",
        "The Bane Chronicles", "Island of Dragons (Unwanteds #7)", "Demigods & Magicians",
        "City of Glass (The Mortal Instruments #3)", "Searching for Meaning in Gailana",
        "A Shard of Ice (The Black Symphony Saga #1)", "King's Folly (The Kinsman Chronicles #1)",
    ],
}
