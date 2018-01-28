from __init__ import *
sys.path.insert(0,'/var/www/vhosts/algotrade.glueckert.net/projects/') 
from SETTINGS.settings_auth import *

mongo_uri = auth['mongo_uri']
client = MongoClient(mongo_uri,waitQueueTimeoutMS=120)
db = client['bitmicro']

atexit.register(exit_handler)

def delta(present,future):
    return future/present

def get_book_df(symbol,min_ts,max_ts, convert_timestamps=False):
    '''
    Returns a DataFrame of book data
    '''
    books_db = db[symbol+'_books']
    query = {'_id': {'$gt': min_ts, '$lt': max_ts}}
    if (min_ts == max_ts == 0):
        query = {}
    cursor = books_db.find(query).sort('_id', pymongo.ASCENDING)
    
    books = pd.DataFrame(list(cursor))
    books = books.set_index('_id')
    if convert_timestamps:
        books.index = pd.to_datetime(books.index, unit='s')

    def to_df(x):
        return pd.DataFrame(x[:10])
    return books.applymap(to_df).sort_index()


def get_width_mid_bid_ask(books):
    '''
    Returns width of best market and midpoint for each data point in DataFrame
    of book data
    '''
    best_bid = books.bids.apply(lambda x: x.price[0])
    best_ask = books.asks.apply(lambda x: x.price[0])
    return best_ask-best_bid, (best_bid + best_ask)/2, best_bid,best_ask


def get_future_mid(books, offset, sensitivity):
    '''
    Returns percent change of future midpoints for each data point in DataFrame
    of book data
    '''
    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity: 
            return books.bid.iloc[i]
        else:
            return(float('nan'))
    change = (books.index.map(future)/books.ask) - 1
    return change

def get_future_mid_all(books, offset, sensitivity):
    '''
    Returns the max percent change of future midpoints for each data point in DataFrame
    of book data
    '''

    def future(timestamp):
        i = books.index.get_loc(timestamp+offset, method='nearest')
        i_now = books.index.get_loc(timestamp, method='nearest')
        if abs(books.index[i] - (timestamp+offset)) < sensitivity:
            return max(books.bid.iloc[i_now:i])
    
    change = (books.index.map(future)/books.ask) - 1
    return change


def get_power_imbalance(books, n=10, power=2):
    '''
    Returns a measure of the imbalance between bids and offers for each data
    point in DataFrame of book data
    '''

    def calc_imbalance(book):
        def calc(x):
            return x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_imbalance = book.bids.iloc[:n].apply(calc, axis=1)
        ask_imbalance = book.asks.iloc[:n].apply(calc, axis=1)
        return (bid_imbalance-ask_imbalance).sum()
    imbalance = books.apply(calc_imbalance, axis=1)
    return imbalance


def get_power_adjusted_price(books, n=10, power=2):
    '''
    Returns the percent change of an average of order prices weighted by inverse
    distance-wieghted volume for each data point in DataFrame of book data
    '''

    def calc_adjusted_price(book):
        def calc(x):
            return x.amount*(.5*book.width/(x.price-book.mid))**power
        bid_inv = 1/book.bids.iloc[:n].apply(calc, axis=1)
        ask_inv = 1/book.asks.iloc[:n].apply(calc, axis=1)
        bid_price = book.bids.price.iloc[:n]
        ask_price = book.asks.price.iloc[:n]
        return (bid_price*bid_inv + ask_price*ask_inv).sum() /\
            (bid_inv + ask_inv).sum()
    adjusted = books.apply(calc_adjusted_price, axis=1)
    return (adjusted/books.mid).apply(log).fillna(0)


def get_trade_df(symbol, min_ts, max_ts, convert_timestamps=False):
    '''
    Returns a DataFrame of trades for symbol in time range
    '''
    trades_db = db[symbol+'_trades']
    query = {'timestamp': {'$gt': min_ts, '$lt': max_ts}}
    if (min_ts == max_ts == 0):
        query = {}
    cursor = trades_db.find(query).sort('_id', pymongo.ASCENDING)
    trades = pd.DataFrame(list(cursor))
    if not trades.empty:
        trades = trades.set_index('_id')
        if convert_timestamps:
            trades.index = pd.to_datetime(trades.index, unit='s')
    return trades


def get_trades_indexes(books, trades, offset, live=False):
    '''
    Returns indexes of trades in offset range for each data point in DataFrame
    of book data
    '''
    def indexes(ts):
        ts = int(ts)
        i_0 = trades.timestamp.searchsorted([ts-offset], side='left')[0]
        if live:
            i_n = -1
        else:
            i_n = trades.timestamp.searchsorted([ts-1], side='right')[0]
        return (i_0, i_n)
    #return books.index.map(indexes)
    return [indexes(ts) for ts in books.index]


def get_trades_count(books, trades):
    '''
    Returns a count of trades for each data point in DataFrame of book data
    '''
    def count(x):
        return len(trades.iloc[x.indexes[0]:x.indexes[1]])
    return books.apply(count, axis=1)


def get_trades_average(books, trades):
    '''
    Returns the percent change of a volume-weighted average of trades for each
    data point in DataFrame of book data
    '''

    def mean_trades(x):
        trades_n = trades.iloc[x.indexes[0]:x.indexes[1]]
        if not trades_n.empty:
            return (trades_n.price*trades_n.amount).sum()/trades_n.amount.sum()
    return (books.mid/books.apply(mean_trades, axis=1)).apply(log).fillna(0)


def get_aggressor(books, trades):
    '''
    Returns a measure of whether trade aggressors were buyers or sellers for
    each data point in DataFrame of book data
    '''

    def aggressor(x):
        trades_n = trades.iloc[x.indexes[0]:x.indexes[1]]
        if trades_n.empty:
            return 0
        buys = trades_n['type'] == 'buy'
        buy_vol = trades_n[buys].amount.sum()
        sell_vol = trades_n[~buys].amount.sum()
        return buy_vol - sell_vol
    return books.apply(aggressor, axis=1)


def get_trend(books, trades):
    '''
    Returns the linear trend in previous trades for each data point in DataFrame
    of book data
    '''

    def trend(x):
        trades_n = trades.iloc[x.indexes[0]:x.indexes[1]]
        if len(trades_n) < 3:
            return 0
        else:
            return linregress(trades_n.index.values, trades_n.price.values)[0]
    return books.apply(trend, axis=1)


def check_times(books):
    '''
    Returns list of differences between collection time and max book timestamps
    for verification purposes
    '''
    time_diff = []
    for i in range(len(books)):
        book = books.iloc[i]
        ask_ts = max(book.asks.timestamp)
        bid_ts = max(book.bids.timestamp)
        ts = max(ask_ts, bid_ts)
        time_diff.append(book.name-ts)
    return time_diff


def make_features(symbol, min_ts,max_ts, mid_offsets,
                  trades_offsets, powers, live=False):
    '''
    Returns a DataFrame with targets and features
    '''
    start = time()
    stage = time()

    # Book related features:
    books = get_book_df(symbol, min_ts,max_ts)
   
    if not live:
        print('get book data run time:', (time()-stage)/60, 'minutes')
        stage = time()
    
    books['width'], books['mid'],books['bid'],books['ask'] = get_width_mid_bid_ask(books)
   
    if not live:
        print('width and mid run time:', (time()-stage)/60, 'minutes')
        stage = time()
    for n in mid_offsets:
        books['mid{}'.format(n)] = get_future_mid(books, n, sensitivity=offset_sensitivity_map(n))
    if not live:
        #----------books = books.dropna()
        print('offset mids run time:', (time()-stage)/60, 'minutes')
        stage = time()
    for p in powers:
        books['imbalance{}'.format(p)] = get_power_imbalance(books, 10, p)
        books['adj_price{}'.format(p)] = get_power_adjusted_price(books, 10, p)
    if not live:
        print('power calcs run time:', (time()-stage)/60, 'minutes')
        stage = time()
    books = books.drop(['bids', 'asks'], axis=1)
    
    # Trade related features:
    min_ts = books.index.min() - trades_offsets[-1]
    max_ts = books.index.max()
    if live:
        max_ts += 10
    trades = get_trade_df(symbol, min_ts, max_ts)
    for n in trades_offsets:
        if trades.empty:
            books['indexes'] = 0
            books['t{}_count'.format(n)] = 0
            books['t{}_av'.format(n)] = 0
            books['agg{}'.format(n)] = 0
            books['trend{}'.format(n)] = 0
        else:
            books['indexes'] = get_trades_indexes(books, trades, n, live)
            books['t{}_count'.format(n)] = get_trades_count(books, trades)
            books['t{}_av'.format(n)] = get_trades_average(books, trades)
            books['agg{}'.format(n)] = get_aggressor(books, trades)
            books['trend{}'.format(n)] = get_trend(books, trades)
    if not live:
        print('trade features run time:', (time()-stage)/60, 'minutes')
        stage = time()
        print('make_features run time:', (time()-start)/60, 'minutes')

    return books.drop('indexes', axis=1)

def get_books_min_max_ts(symbol):
    books_db = db[symbol+'_books']
    cursor = books_db.find({}).sort('_id', pymongo.ASCENDING).limit(1)
    books = pd.DataFrame(list(cursor))
    books = books.set_index('_id')
    min_ts=books.index[0]
    cursor = books_db.find({}).sort('_id', pymongo.DESCENDING).limit(1)
    books = pd.DataFrame(list(cursor))
    books = books.set_index('_id')
    max_ts=books.index[0]
    return {'min_ts' : min_ts,'max_ts' : max_ts, 'days' : ((max_ts - min_ts) / 24 / 3600),
           'min_date' : pd.to_datetime(min_ts,unit='s').date(),
            'max_date' : pd.to_datetime(max_ts,unit='s').date()}

#----find full days
def get_books_days_ts(symbol,date_from='',date_to=''):
    dates_books=get_books_min_max_ts("btc")
    dates_range = pd.date_range(dates_books['min_date'],dates_books['max_date'])
    dates_range = {d.date() : [dateparse(str(d.date()) + ' 00:00:01').timestamp(),dateparse(str(d.date()) + ' 23:59:59').timestamp()] for d in dates_range}
    dates_range[dates_books['min_date']][0] = dates_books['min_ts']
    dates_range[dates_books['max_date']][1] = dates_books['max_ts']
    od = collections.OrderedDict(sorted(dates_range.items()))
    booklist = []
    for key, value in od.items():booklist.append(dict(date=key,symbol=symbol,start=value[0],end=value[1]))
    booklist = [ e for e in booklist if e['date'] >= date_from and e['date'] <= date_to]
    return booklist
    
def merge_csvs(path):
    import glob
    # get data file names
    filenames = glob.glob(path + "*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename))

    # Concatenate all data into one DataFrame
    d = pd.concat(dfs, ignore_index=True)
    d = d.rename(columns={'_id': 'timstamp'})
    d=d.drop_duplicates(subset='timstamp')
    d=d.sort_values(by='timstamp')
    return d


def offset_sensitivity_map(offset):
    if offset >= 0 & offset < 200:
        s = 1
    if offset >= 200 & offset < 300:
        s = 2
    if offset >=300 & offset < 600:
        s = 5
    if offset >=600:
        s = 10
    return s
